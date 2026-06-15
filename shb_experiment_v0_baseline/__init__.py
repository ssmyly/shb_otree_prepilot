"""
shb_experiment/__init__.py
==========================
oTree 5 implementation of the Salary History Ban (SHB) Experiment.

Design:
  - Between-subjects: SHB condition vs. No-SHB condition
  - 3 rounds of real-effort number-counting task
  - Investment stage (training tokens) before each round
  - Bayesian wage-setting formula (calibrated to model parameters)

Model reference: shb_model.tex (Equations 3-4, Appendix B)
Design reference: experiment/design/experiment_design.md

Usage:
  otree devserver        (development)
  otree prodserver       (production, set OTREE_PRODUCTION=1)

Authors: [Author]
"""

from otree.api import *
import random
import json
import math


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

doc = """
Salary History Ban Experiment

Workers complete 3 rounds of a number-grid counting task.
Before each round they may purchase training tokens (human capital investment).
Wages are set by a Bayesian formula that either uses full salary history
(No-SHB condition) or only the current round's signal (SHB condition).

Primary hypothesis: training investment is higher in No-SHB condition.
"""


class C(BaseConstants):
    NAME_IN_URL = 'shb'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 3

    # ── Task parameters ──────────────────────────────────────
    TASK_DURATION_SECONDS = 120    # 2 minutes per round
    BASELINE_DURATION_SECONDS = 60 # 1 minute calibration round
    GRID_ROWS = 5
    GRID_COLS = 6
    # Baseline uses varying grid sizes to noisily measure productivity
    BASELINE_GRID_SIZES = [(3, 4), (4, 5), (5, 6), (5, 7), (6, 7)]

    # ── Investment parameters ─────────────────────────────────
    MAX_TRAINING_TOKENS = 3
    TOKEN_COST_ECU = 5             # cost per training token in ECU
    TRAINING_BONUS_PER_TOKEN = 5   # score bonus per token (points)
    SCORE_MULTIPLIER = 5           # raw correct answers × 5 → score

    # ── Signal parameters ─────────────────────────────────────
    NOISE_MIN = -10                # uniform noise lower bound
    NOISE_MAX = 10                 # uniform noise upper bound
    THRESHOLD = 55                 # signal must be ≥ this to be "hired"

    # ── Bayesian wage-setting parameters ─────────────────────
    # Calibrated to: θ ~ N(50, 200), ε ~ N(0, 300)
    # κ_n = σ_θ² / (σ_θ² + σ²/n) = 200 / (200 + 300/n)
    PRIOR_MEAN = 50.0              # μ_θ
    PRIOR_VAR = 200.0              # σ_θ²
    NOISE_VAR = 300.0              # σ²

    # Pre-computed Kalman gains for rounds 1, 2, 3
    # κ_1 = 200/500 = 0.400
    # κ_2 = 200/350 ≈ 0.5714
    # κ_3 = 200/300 ≈ 0.6667
    KAPPA = [0.400, 0.5714, 0.6667]

    # ── Payment ───────────────────────────────────────────────
    ECU_PER_USD = 10               # 10 ECU = $1
    PARTICIPATION_FEE_ECU = 30     # $3.00 base pay in ECU


# ─────────────────────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────────────────────

class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


def creating_session(subsession: Subsession):
    """
    Assign conditions on first round only (module-level — oTree 5 style).
    Balanced random assignment (1:1 SHB / No-SHB).
    Propagates condition to player.condition each round.
    """
    if subsession.round_number == 1:
        players = subsession.get_players()
        n = len(players)
        conditions = (['shb'] * (n // 2) + ['no_shb'] * (n - n // 2))
        random.shuffle(conditions)
        for player, cond in zip(players, conditions):
            player.participant.vars['condition'] = cond
            player.condition = cond
    else:
        for player in subsession.get_players():
            player.condition = player.participant.vars.get('condition', 'shb')


class Player(BasePlayer):

    # ── Assignment ────────────────────────────────────────────
    condition = models.StringField()   # 'shb' or 'no_shb'

    # ── Baseline calibration (round 1 only) ───────────────────
    # Worker performs the task once; we compute a private signal and a
    # noisy wage offer. The worker is shown ONLY the wage offer.
    baseline_num_correct = models.IntegerField(initial=0)
    baseline_score = models.IntegerField(initial=0)         # private to researcher
    baseline_noise = models.IntegerField(initial=0)         # private
    baseline_signal = models.IntegerField(initial=0)        # private
    baseline_wage_offer = models.FloatField(initial=0.0)    # SHOWN to worker
    baseline_hired = models.BooleanField(initial=False)     # SHOWN to worker
    baseline_data_json = models.LongStringField(initial='[]')

    # ── Investment decision ───────────────────────────────────
    training_tokens = models.IntegerField(
        label="How many training tokens would you like to purchase?",
        min=0,
        max=C.MAX_TRAINING_TOKENS,
    )

    # ── Task outcomes ─────────────────────────────────────────
    num_correct = models.IntegerField(initial=0)
    task_score = models.IntegerField(initial=0)    # num_correct × SCORE_MULTIPLIER
    training_bonus = models.IntegerField(initial=0)  # tokens × TRAINING_BONUS_PER_TOKEN
    raw_score = models.IntegerField(initial=0)     # task_score + training_bonus
    noise_drawn = models.IntegerField(initial=0)   # ε sampled from Uniform[NOISE_MIN, NOISE_MAX]
    signal = models.IntegerField(initial=0)        # raw_score + noise_drawn

    # ── Hiring and wages ──────────────────────────────────────
    hired = models.BooleanField(initial=False)
    wage = models.FloatField(initial=0.0)          # ECU
    kappa_used = models.FloatField(initial=0.0)    # Kalman gain applied this round
    mean_signal_used = models.FloatField(initial=0.0)  # mean of signals entering wage formula

    # ── Financials ────────────────────────────────────────────
    training_cost = models.FloatField(initial=0.0)
    round_net = models.FloatField(initial=0.0)     # wage - training_cost (pre-floor)

    # ── Task data (serialised JSON for debugging / analysis) ──
    task_data_json = models.LongStringField(initial='[]')

    # ── Survey fields ─────────────────────────────────────────
    survey_strategy = models.LongStringField(
        label="How did you decide how many training tokens to buy each round?",
        blank=True,
    )
    survey_investment_trend = models.StringField(
        label="Did you try to invest more as the study progressed?",
        choices=['Yes', 'No', 'No particular pattern'],
        blank=True,
    )
    survey_salary_history_belief = models.IntegerField(
        label=(
            "In real life, do you believe your current salary affects wage offers "
            "from new employers? (1 = Not at all, 5 = Strongly)"
        ),
        min=1, max=5,
        blank=True,
    )
    survey_shb_awareness = models.StringField(
        label="Are you aware of laws that ban employers from asking about salary history?",
        choices=['Yes', 'No', 'Not sure'],
        blank=True,
    )
    survey_shb_support = models.StringField(
        label="Do you support laws that ban employers from asking about salary history?",
        choices=['Support', 'Oppose', 'Neither support nor oppose'],
        blank=True,
    )
    survey_gender = models.StringField(
        label="Gender",
        choices=['Male', 'Female', 'Non-binary / third gender', 'Prefer not to say'],
        blank=True,
    )
    survey_age = models.IntegerField(label="Age", min=18, max=100, blank=True)
    survey_education = models.StringField(
        label="Highest level of education completed",
        choices=[
            'Less than high school',
            'High school diploma or GED',
            'Some college, no degree',
            'Associate degree',
            "Bachelor's degree",
            "Master's degree",
            'Doctoral or professional degree',
        ],
        blank=True,
    )
    survey_employment = models.StringField(
        label="Current employment status",
        choices=['Full-time employed', 'Part-time employed', 'Both (multiple jobs)'],
        blank=True,
    )
    survey_income = models.StringField(
        label="Annual household income",
        choices=[
            'Under $30,000',
            '$30,000 – $49,999',
            '$50,000 – $74,999',
            '$75,000 – $99,999',
            '$100,000 – $149,999',
            '$150,000 or more',
            'Prefer not to say',
        ],
        blank=True,
    )
    survey_job_search_freq = models.StringField(
        label="How often do you search for new jobs?",
        choices=['Never', 'Rarely (less than once a year)', 'Sometimes (once a year)', 'Often (multiple times per year)'],
        blank=True,
    )

    # ── Comprehension check ───────────────────────────────────
    comp_check_tokens = models.StringField(
        label="What did training tokens do in this study?",
        choices=[
            'They added bonus points to my score',
            'They gave me hints during the task',
            'They had no effect on my score',
            'They changed the wage formula',
        ],
        blank=True,
    )
    comp_check_info = models.StringField(
        label="Under your condition, what did the employer use to determine your wage?",
        blank=True,
    )


# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_condition(player: Player) -> str:
    """
    Defensive accessor: returns the player's condition, assigning one if
    creating_session somehow didn't run (e.g. demo URL edge cases).
    """
    cond = player.field_maybe_none('condition')
    if cond is None:
        cond = player.participant.vars.get('condition')
        if cond is None:
            cond = random.choice(['shb', 'no_shb'])
            player.participant.vars['condition'] = cond
        player.condition = cond
    return cond


def compute_wage(player: Player) -> tuple[float, float, float]:
    """
    Compute the wage for this round under the player's condition.

    Returns:
        (wage, kappa_used, mean_signal_used)

    Wage formula:
      SHB:    w = prior_mean + κ₁ × (signal_t - prior_mean)
      No-SHB: w = prior_mean + κₙ × (mean(all_signals) - prior_mean)

    where κₙ = σ_θ² / (σ_θ² + σ²/n) is the Kalman gain with n signals.
    Reference: shb_model.tex, Equations (3)–(4).
    """
    prior_mean = C.PRIOR_MEAN
    prior_var = C.PRIOR_VAR
    noise_var = C.NOISE_VAR

    if get_condition(player) == 'no_shb':
        # Collect all signals this player has generated up to and including now
        past_signals = [
            p.signal for p in player.in_previous_rounds()
        ]
        all_signals = past_signals + [player.signal]
        n = len(all_signals)
        mean_sig = sum(all_signals) / n
        # Kalman gain with n signals
        posterior_precision = 1.0 / prior_var + n / noise_var
        kappa_n = (n / noise_var) / posterior_precision  # = σ_θ² / (σ_θ² + σ²/n)
    else:
        # SHB: only current signal
        all_signals = [player.signal]
        n = 1
        mean_sig = player.signal
        posterior_precision = 1.0 / prior_var + 1.0 / noise_var
        kappa_n = (1.0 / noise_var) / posterior_precision

    wage = prior_mean + kappa_n * (mean_sig - prior_mean)
    return (round(wage, 2), round(kappa_n, 4), round(mean_sig, 2))


def get_salary_history(player: Player) -> list[dict]:
    """
    Return a list of dicts with historical round data for display.
    Used in Investment and Results pages under No-SHB condition.
    """
    history = []
    for pr in player.in_previous_rounds():
        history.append({
            'round': pr.round_number,
            'tokens': pr.training_tokens,
            'signal': pr.signal,
            'hired': pr.hired,
            'wage': round(pr.wage, 2),
        })
    return history


def format_currency(val: float) -> str:
    return f"{val:.2f}"


# ─────────────────────────────────────────────────────────────
#  PAGES
# ─────────────────────────────────────────────────────────────

class Consent(Page):
    """Informed consent + welcome screen."""

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player: Player):
        return {
            'participation_fee_usd': C.PARTICIPATION_FEE_ECU / C.ECU_PER_USD,
            'ecu_per_usd': C.ECU_PER_USD,
            'num_rounds': C.NUM_ROUNDS,
        }


class Instructions(Page):
    """Full task instructions. Displayed once at start."""

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player: Player):
        condition = get_condition(player)
        is_shb = (condition == 'shb')
        # Pre-compute example wages for instructions
        example_signal = 65
        example_w_shb = C.PRIOR_MEAN + C.KAPPA[0] * (example_signal - C.PRIOR_MEAN)
        example_w_noshb_r2 = C.PRIOR_MEAN + C.KAPPA[1] * (example_signal - C.PRIOR_MEAN)
        # Demo grid (5 rows × 6 cols) for the instructions illustration
        demo_grid = [3, 7, 2, 9, 1, 4, 0, 5, 7, 8, 2, 6,
                     1, 4, 7, 3, 6, 0, 9, 2, 1, 4, 7, 5,
                     0, 8, 3, 6, 2, 1]
        demo_cells = [{'digit': d, 'is_target': (d == 7)} for d in demo_grid]

        return {
            'condition': condition,
            'is_shb': is_shb,
            'token_cost': C.TOKEN_COST_ECU,
            'training_bonus': C.TRAINING_BONUS_PER_TOKEN,
            'max_tokens': C.MAX_TRAINING_TOKENS,
            'max_token_cost': C.TOKEN_COST_ECU * C.MAX_TRAINING_TOKENS,
            'max_token_bonus': C.TRAINING_BONUS_PER_TOKEN * C.MAX_TRAINING_TOKENS,
            'token_cost_2': C.TOKEN_COST_ECU * 2,
            'token_bonus_2': C.TRAINING_BONUS_PER_TOKEN * 2,
            'demo_cells': demo_cells,
            'threshold': C.THRESHOLD,
            'noise_min': C.NOISE_MIN,
            'noise_max': C.NOISE_MAX,
            'num_rounds': C.NUM_ROUNDS,
            'task_duration': C.TASK_DURATION_SECONDS,
            'score_multiplier': C.SCORE_MULTIPLIER,
            'kappa_1': C.KAPPA[0],
            'kappa_2': C.KAPPA[1],
            'kappa_3': C.KAPPA[2],
            'prior_mean': int(C.PRIOR_MEAN),
            'example_signal': example_signal,
            'example_w_shb': round(example_w_shb, 1),
            'example_w_noshb_r2': round(example_w_noshb_r2, 1),
            'participation_fee_usd': C.PARTICIPATION_FEE_ECU / C.ECU_PER_USD,
            'ecu_per_usd': C.ECU_PER_USD,
        }


class Baseline(Page):
    """
    Calibration round (round 1 only).

    Worker performs a brief counting task with varying grid sizes.
    The system computes a private signal (score + noise) and a single-signal
    Bayesian wage offer using κ₁. The worker is shown ONLY the wage offer
    (and hired/not-hired) — NOT the score, NOT the noise, NOT the raw signal.

    This mirrors a real labor market where the worker has some prior signal
    of their own productivity (via a wage offer) but doesn't directly observe
    the employer's underlying assessment.
    """

    form_model = 'player'
    form_fields = ['baseline_num_correct', 'baseline_data_json']

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def get_timeout_seconds(player: Player):
        return C.BASELINE_DURATION_SECONDS

    @staticmethod
    def vars_for_template(player: Player):
        return {
            'task_duration': C.BASELINE_DURATION_SECONDS,
            'grid_sizes_json': json.dumps(C.BASELINE_GRID_SIZES),
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        # Compute private baseline signal
        player.baseline_score = player.baseline_num_correct * C.SCORE_MULTIPLIER
        player.baseline_noise = random.randint(C.NOISE_MIN, C.NOISE_MAX)
        player.baseline_signal = player.baseline_score + player.baseline_noise

        # Wage offer using single-signal Bayesian formula (κ₁)
        if player.baseline_signal >= C.THRESHOLD:
            player.baseline_hired = True
            player.baseline_wage_offer = round(
                C.PRIOR_MEAN + C.KAPPA[0] * (player.baseline_signal - C.PRIOR_MEAN), 2
            )
        else:
            player.baseline_hired = False
            player.baseline_wage_offer = 0.0

        # Persist on participant for retrieval in later rounds (e.g. for
        # No-SHB wage history if you decide to include the baseline signal)
        player.participant.vars['baseline_signal'] = player.baseline_signal
        player.participant.vars['baseline_wage_offer'] = player.baseline_wage_offer
        player.participant.vars['baseline_hired'] = player.baseline_hired


class BaselineResult(Page):
    """
    Show the worker ONLY the noisy wage offer from baseline.
    No score. No signal. No noise. Just: "Based on the calibration round,
    your wage offer would be $X (or: you would not be hired)."
    """

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player: Player):
        return {
            'wage_offer': player.baseline_wage_offer,
            'hired': player.baseline_hired,
            'is_shb': (get_condition(player) == 'shb'),
            'threshold': C.THRESHOLD,
            'prior_mean': int(C.PRIOR_MEAN),
        }


class Investment(Page):
    """
    Investment decision stage.
    Participant chooses training_tokens (0–3).
    No-SHB participants see their salary history.
    """

    form_model = 'player'
    form_fields = ['training_tokens']

    @staticmethod
    def vars_for_template(player: Player):
        condition = get_condition(player)
        is_shb = (condition == 'shb')
        history = get_salary_history(player)
        kappa_next = C.KAPPA[min(player.round_number - 1, len(C.KAPPA) - 1)]

        # Show cost–benefit table
        token_table = []
        for t in range(C.MAX_TRAINING_TOKENS + 1):
            token_table.append({
                'tokens': t,
                'cost': t * C.TOKEN_COST_ECU,
                'bonus': t * C.TRAINING_BONUS_PER_TOKEN,
            })

        return {
            'round_number': player.round_number,
            'num_rounds': C.NUM_ROUNDS,
            'is_shb': is_shb,
            'condition': condition,
            'history': history,
            'history_count_plus_1': len(history) + 1,
            'token_table': token_table,
            'token_cost': C.TOKEN_COST_ECU,
            'max_tokens': C.MAX_TRAINING_TOKENS,
            'training_bonus': C.TRAINING_BONUS_PER_TOKEN,
            'threshold': C.THRESHOLD,
            'noise_min': C.NOISE_MIN,
            'noise_max': C.NOISE_MAX,
            'kappa_next': round(kappa_next, 3),
            'prior_mean': int(C.PRIOR_MEAN),
            'has_history': (len(history) > 0),
        }


class Task(Page):
    """
    Real-effort number-counting task (2 minutes).
    Grid and problem generation handled in JavaScript (Task.html).
    num_correct is written to a hidden field by JS on timeout/submit.
    """

    form_model = 'player'
    form_fields = ['num_correct', 'task_data_json']

    @staticmethod
    def get_timeout_seconds(player: Player):
        return C.TASK_DURATION_SECONDS

    @staticmethod
    def vars_for_template(player: Player):
        return {
            'round_number': player.round_number,
            'num_rounds': C.NUM_ROUNDS,
            'training_tokens': player.training_tokens,
            'training_bonus_total': player.training_tokens * C.TRAINING_BONUS_PER_TOKEN,
            'task_duration': C.TASK_DURATION_SECONDS,
            'threshold': C.THRESHOLD,
            'noise_min': C.NOISE_MIN,
            'noise_max': C.NOISE_MAX,
            'grid_rows': C.GRID_ROWS,
            'grid_cols': C.GRID_COLS,
            'score_multiplier': C.SCORE_MULTIPLIER,
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        """
        After task: compute scores, draw noise, set signal, compute wage.
        """
        # 1. Compute scores
        player.task_score = player.num_correct * C.SCORE_MULTIPLIER
        player.training_bonus = player.training_tokens * C.TRAINING_BONUS_PER_TOKEN
        player.raw_score = player.task_score + player.training_bonus

        # 2. Draw noise and compute signal
        player.noise_drawn = random.randint(C.NOISE_MIN, C.NOISE_MAX)
        player.signal = player.raw_score + player.noise_drawn

        # 3. Determine hiring
        player.hired = (player.signal >= C.THRESHOLD)

        # 4. Compute wage
        if player.hired:
            wage, kappa, mean_sig = compute_wage(player)
            player.wage = max(0.0, wage)
            player.kappa_used = kappa
            player.mean_signal_used = mean_sig
        else:
            player.wage = 0.0
            player.kappa_used = 0.0
            player.mean_signal_used = 0.0

        # 5. Compute net earnings this round
        player.training_cost = float(player.training_tokens * C.TOKEN_COST_ECU)
        player.round_net = player.wage - player.training_cost

        # 6. Set oTree payoff (floor at 0; participation fee added at end)
        player.payoff = max(0, player.round_net)


class Results(Page):
    """
    Round results page.
    Shows score breakdown, signal, hiring outcome, wage, and history (No-SHB only).
    """

    @staticmethod
    def vars_for_template(player: Player):
        condition = get_condition(player)
        is_shb = (condition == 'shb')
        history = get_salary_history(player)
        # Add current round to history for display
        history_with_current = history + [{
            'round': player.round_number,
            'tokens': player.training_tokens,
            'signal': player.signal,
            'hired': player.hired,
            'wage': round(player.wage, 2),
        }]

        # Wage decomposition for display
        if player.hired and not is_shb:
            wage_from_current = C.PRIOR_MEAN + C.KAPPA[0] * (player.signal - C.PRIOR_MEAN)
            history_premium = player.wage - wage_from_current
        else:
            wage_from_current = player.wage
            history_premium = 0.0

        return {
            'round_number': player.round_number,
            'num_rounds': C.NUM_ROUNDS,
            'is_shb': is_shb,
            'condition': condition,
            'training_tokens': player.training_tokens,
            'training_cost': player.training_cost,
            'num_correct': player.num_correct,
            'task_score': player.task_score,
            'training_bonus': player.training_bonus,
            'raw_score': player.raw_score,
            'noise_drawn': player.noise_drawn,
            'signal': player.signal,
            'hired': player.hired,
            'wage': round(player.wage, 2),
            'kappa_used': round(player.kappa_used, 3),
            'mean_signal_used': round(player.mean_signal_used, 2),
            'round_net': round(player.round_net, 2),
            'net_floored': round(max(0, player.round_net), 2),
            'threshold': C.THRESHOLD,
            'prior_mean': int(C.PRIOR_MEAN),
            'score_multiplier': C.SCORE_MULTIPLIER,
            'history': history_with_current,
            'has_history': (len(history_with_current) > 1),
            'wage_from_current': round(wage_from_current, 2),
            'history_premium': round(history_premium, 2),
            'is_last_round': (player.round_number == C.NUM_ROUNDS),
        }


class FinalResults(Page):
    """
    Summary of all 3 rounds, total earnings, and payment information.
    """

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player: Player):
        all_rounds = player.in_all_rounds()

        history = []
        for r in all_rounds:
            history.append({
                'round': r.round_number,
                'tokens': r.training_tokens,
                'cost': round(r.training_cost, 2),
                'correct': r.num_correct,
                'task_score': r.task_score,
                'bonus': r.training_bonus,
                'raw_score': r.raw_score,
                'noise': r.noise_drawn,
                'signal': r.signal,
                'hired': r.hired,
                'wage': round(r.wage, 2),
                'net': round(max(0.0, r.round_net), 2),
            })

        total_wages = sum(r.wage for r in all_rounds)
        total_costs = sum(r.training_cost for r in all_rounds)
        total_correct = sum(r.num_correct for r in all_rounds)
        total_tokens = sum(r.training_tokens for r in all_rounds)
        times_hired = sum(1 for r in all_rounds if r.hired)

        # Total ECU = sum of floored round nets + participation fee
        total_bonus_ecu = sum(max(0.0, r.round_net) for r in all_rounds)
        total_ecu = total_bonus_ecu + C.PARTICIPATION_FEE_ECU
        total_usd = total_ecu / C.ECU_PER_USD

        return {
            'history': history,
            'total_wages': round(total_wages, 2),
            'total_costs': round(total_costs, 2),
            'total_correct': total_correct,
            'total_tokens': total_tokens,
            'times_hired': times_hired,
            'total_bonus_ecu': round(total_bonus_ecu, 2),
            'participation_fee_ecu': C.PARTICIPATION_FEE_ECU,
            'total_ecu': round(total_ecu, 2),
            'total_usd': round(total_usd, 2),
            'bonus_usd': round(total_bonus_ecu / C.ECU_PER_USD, 2),
            'participation_usd': C.PARTICIPATION_FEE_ECU / C.ECU_PER_USD,
            'ecu_per_usd': C.ECU_PER_USD,
            'condition': get_condition(player),
            'is_shb': (get_condition(player) == 'shb'),
            'num_rounds': C.NUM_ROUNDS,
        }


class Survey(Page):
    """
    Post-task survey: comprehension check, strategy, beliefs, demographics.
    """

    form_model = 'player'
    form_fields = [
        'comp_check_tokens',
        'comp_check_info',
        'survey_strategy',
        'survey_investment_trend',
        'survey_salary_history_belief',
        'survey_shb_awareness',
        'survey_shb_support',
        'survey_gender',
        'survey_age',
        'survey_education',
        'survey_employment',
        'survey_income',
        'survey_job_search_freq',
    ]

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player: Player):
        condition = get_condition(player)
        is_shb = (condition == 'shb')
        return {
            'is_shb': is_shb,
            'condition': condition,
            'correct_comp_answer': (
                'They added bonus points to my score'
            ),
            'correct_info_answer': (
                'Only your performance in this round'
                if is_shb else
                'Your performance in this round AND your full salary history'
            ),
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        """
        Set final payoff on the last page.
        oTree accumulates payoffs across rounds automatically;
        we add the participation fee here.
        """
        if player.round_number == C.NUM_ROUNDS:
            player.participant.payoff += C.PARTICIPATION_FEE_ECU


# ─────────────────────────────────────────────────────────────
#  PAGE SEQUENCE
# ─────────────────────────────────────────────────────────────

page_sequence = [
    Consent,
    Instructions,
    Baseline,         # round 1 only — calibration
    BaselineResult,   # round 1 only — show wage offer (private signal)
    Investment,
    Task,
    Results,
    FinalResults,
    Survey,
]
