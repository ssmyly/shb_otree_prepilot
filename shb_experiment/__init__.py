"""
shb_experiment/__init__.py
==========================
oTree 5 implementation of the Salary History Ban (SHB) Experiment.

Design (see experiment/design/design_decisions.tex):
  - Between-subjects: SHB condition vs. No-SHB condition.
  - Currency shown to participants is "points" (1 point = $0.10 in the
    payout conversion handled by oTree's real_world_currency_per_point).
  - 3 oTree rounds mapping onto model periods:
      Round 1 = t=0 (calibration) -- counting task only, no investment;
                worker shown only the wage offer w_0 (not signal/score/noise).
      Round 2 = t=1 (first work period) -- invest, task, signal, wage.
      Round 3 = t=2 (second/terminal work period) -- invest, task, signal, wage.
  - Wage rule (Decision D2):
      SHB  : w_t = mu + kappa_1 (s_t - mu)
      No-SHB: w_t = mu + kappa_1 (s_t - mu) + beta * sum_{j<t} (w_j - mu)

Model reference: model_codex/general_sequential_model.pdf
Design reference: experiment/design/design_decisions.tex

Usage:
  otree devserver        (development)
  otree prodserver       (production, set OTREE_PRODUCTION=1)
"""

from otree.api import *
import random
import json
import math
import time


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

doc = """
Salary History Ban Experiment

Workers complete 1 calibration round + 2 work rounds of a number-grid counting
task, preceded by an unpaid practice round. Before each work round they may
purchase training tokens (human capital investment) which shrink the counting
grid. Wages are set by a stipulated formula that either uses past wages
(No-SHB condition) or only the current round's performance signal (SHB
condition).

Primary hypothesis: training investment is higher under No-SHB, with the
gap largest for participants with favorable past wages (Corollary 1).
"""


class C(BaseConstants):
    NAME_IN_URL = 'shb_exp'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 3

    # ── Task parameters ──────────────────────────────────────
    # 2026-05-03: durations equalized across Practice, Baseline (calibration),
    # and the work rounds so that the raw score scale (num_correct ×
    # SCORE_MULTIPLIER) is comparable across rounds. Previously calibration
    # was 60s while work rounds were 120s, which inflated the work-round
    # signals by a factor of ~2 for any given productivity rate and broke
    # cross-round comparability of μ, κ₁, β, and the thresholds.
    TASK_DURATION_SECONDS = 90        # 1.5 minutes per work round
    BASELINE_DURATION_SECONDS = 90    # 1.5 minutes calibration round
    # Decision H (H1, 2026-04-27): unpaid practice round before calibration.
    # Pure familiarisation; performance is recorded for analysis (round-covariate
    # learning fit) but does not affect any wage offer or payment. Length
    # equalized to 90s on 2026-05-03 to match the rounds it warms up for.
    PRACTICE_DURATION_SECONDS = 90
    # Default grid for the calibration round (no investment in t=0)
    GRID_ROWS = 5
    GRID_COLS = 6
    # Baseline uses varying grid sizes to noisily measure productivity
    BASELINE_GRID_SIZES = [(3, 4), (4, 5), (5, 6)]

    # ── Investment parameters ─────────────────────────────────
    # Decision B (2026-04-27): investment reduces task complexity by shrinking
    # the grid. Skill no longer adds points; it makes the task easier so the
    # worker can complete more cells in fixed time. Mapping (tokens → rows×cols):
    #   0 → 5×6 (30 cells, baseline difficulty)
    #   1 → 5×5 (25 cells, ~17% smaller)
    #   2 → 4×5 (20 cells, ~33% smaller)
    #   3 → 4×4 (16 cells, ~47% smaller)
    # Decision C (2026-04-27; re-tuned 2026-06-03): cumulative cost is convex
    # (marginal cost rises 10 → 14 → 24 points) so the privately optimal
    # investment varies across workers and treatment arms; cost is stationary
    # across rounds.
    #
    # 2026-06-03 re-tuning (Harrison 1989, "Theory and Misbehavior of First-
    # Price Auctions"): evaluate behavior in EXPECTED-PAYOFF space, not action
    # space. Under the previous schedule {0,4,10,18} the grid-shrink
    # productivity benefit dominated the token cost, so net payoff rose
    # monotonically in tokens — the 4×4 grid was BOTH the easiest task and the
    # money-maximizing choice. Foregone income from "always buy 3" was ≤ 0,
    # far below any plausible perceptive threshold (Harrison uses 1–25¢), so
    # the SHB→investment channel was unidentifiable: every subject corners at
    # 3 regardless of treatment. The schedule below (in points; dollar values
    # at POINTS_PER_USD=20 are $0.00 / $0.50 / $1.20 / $2.40) makes the optimum
    # interior (~2 tokens for the median worker) and puts a salient ≈$0.33
    # expected-payoff wedge on the 2→3 step (above the conservative end of
    # Harrison's threshold), so over-investing for task-ease is now a
    # perceptible monetary sacrifice rather than a free choice, and the No-SHB
    # wage-history premium can shift workers across an interior margin. The
    # token price is borne out of a fungible starting allowance + earned wages
    # (see TOKEN_ENDOWMENT_POINTS), so the cost is real take-home money foregone.
    MAX_TRAINING_TOKENS = 3
    GRID_BY_TOKENS = {
        0: (5, 6),
        1: (5, 5),
        2: (4, 5),
        3: (4, 4),
    }
    COST_BY_TOKENS = {
        0: 0,
        1: 10,
        2: 24,
        3: 48,
    }
    SCORE_MULTIPLIER = 5           # raw correct answers × 5 → score

    # ── Signal parameters ─────────────────────────────────────
    # Decision E (E1, 2026-04-27): ε_t drawn uniformly on integers in
    # [-NOISE_Q, NOISE_Q]. Compactly supported (A1 of Section III holds).
    # This matches the uniform-linear specialization the Section III
    # closed-form footnotes use; Section V's "normally-distributed noise"
    # phrasing will be revised at the paper-rewrite step.
    NOISE_Q = 10                   # ε_t ∈ {-NOISE_Q, ..., NOISE_Q} (discrete uniform)
    NOISE_MIN = -NOISE_Q
    NOISE_MAX = NOISE_Q
    # Decision F (F3, 2026-04-27): retention threshold differs by period.
    # r_0 = 45 for the calibration round so almost every participant has an
    # observed w_0 (the heterogeneity test requires variation in observed w_0).
    # r_1 = r_2 = 55 for the work rounds preserves the threshold bite where
    # Channel III's investment-to-retain incentive lives. Pilot data may
    # re-calibrate either value.
    THRESHOLD_BASELINE = 45        # r_0 (calibration round)
    THRESHOLD_WORK = 55            # r_1 = r_2 (work rounds)

    # ── Wage-setting parameters ──────────────────────────────
    # Decision D (D2, 2026-04-27): wage-history additive form.
    #   SHB:  w_t = μ + κ₁ (s_t - μ)
    #   NB :  w_t = μ + κ₁ (s_t - μ) + β · Σ_{j<t} (w_j - μ)
    # κ₁ = 0.40 is a chosen coefficient. (Genealogy: it equals the Gaussian
    # Bayesian gain σ_θ² / (σ_θ² + σ²) = 200/500 = 0.40 that one would compute
    # under θ ~ N(50, 200), ε ~ N(0, 300). Under E1's uniform noise this is no
    # longer a Bayesian primitive; the constant is retained because it gives
    # signals a sensible weight in the wage and dovetails with the Section III
    # uniform-linear closed forms.)
    # β controls how strongly past wages enter the current wage under NB.
    # See design_decisions.tex Decisions D and E for derivation and A1–A8 check.
    PRIOR_MEAN = 50.0              # μ
    KAPPA_1 = 0.40                 # weight on (s_t - μ) under both regimes
    HISTORY_BETA = 0.5             # weight on each past (w_j - μ) under NB only

    # ── Payment ───────────────────────────────────────────────
    # Participants see all monetary amounts in "points" (settings.py sets
    # real_world_currency_per_point = 1/POINTS_PER_USD, so 1 point = $1/20 = $0.05).
    #
    # 2026-06-03 re-tuning (Harrison 1989): the conversion is the level lever,
    # not the gradient lever. Cutting it cheapens the whole payout but also
    # FLATTENS expected-payoff space — exactly the failure Harrison warns
    # against — so it cannot be cut without re-steepening the token-cost
    # schedule above. PPU was raised 10 → 20 (1 pt: $0.10 → $0.05), which
    # roughly halves the per-point value while keeping the three paid rounds'
    # wages capping near ~$8 of game earnings; pushing further (e.g. PPU=40)
    # flattens the gradient enough that the interior token optimum collapses
    # back to 0. PPU=20 is where the interior optimum survives.
    POINTS_PER_USD = 20            # 20 points = $1  (1 point = $0.05)

    # ── Payout structure (fast players' ceiling ≈ $13 per participant) ────
    # 2026-06-10: base pay REMOVED. The guaranteed $3 base was deleted and its
    # value folded into the allowance ($2 → $5) per a design decision to make
    # pay effort-leaning (there is no longer a separate guaranteed component).
    #   $5 token allowance   (TOKEN_ENDOWMENT_POINTS, fungible: kept if unspent)
    # + up to ~$8 game wages (earned across the 1 calibration + 2 work rounds)
    #
    # The token allowance is FUNGIBLE: any of it not spent on tokens is paid
    # out as take-home cash, and a participant who spends it on tokens and is
    # then not hired can finish near $0 — there is NO guaranteed minimum.
    # Fungibility is also essential under Harrison — if the allowance were
    # use-it-or-lose-it, tokens would be free up to $5 and the 4×4 grid would
    # again be a costless choice, reintroducing the flat-maximum problem.
    # Because unspent allowance is real money, the true cost of a token is the
    # cash foregone, so the expected-payoff gradient above is honest. Tokens are
    # paid out of (allowance + wages earned so far); the budget constraint
    # (compute_bank_balance) blocks unaffordable purchases.
    BASE_PAY_POINTS = 0            # base pay removed (was 60 = $3.00)
    TOKEN_ENDOWMENT_POINTS = 100   # $5.00 starting token allowance (fungible)
    # Back-compat alias: several code paths still read PARTICIPATION_FEE_POINTS.
    # Keep it pointing at the (now-zero) base component.
    PARTICIPATION_FEE_POINTS = BASE_PAY_POINTS

    # ── Dollar-denominated display values ────────────────────────
    # Signal threshold converts to a wage threshold via w = μ + κ₁(s − μ),
    # so signal ≥ r  ⟺  wage ≥ μ + κ₁(r − μ).
    # All participant-facing displays use these dollar equivalents; the
    # underlying signal/points scale is never shown to participants.
    THRESHOLD_BASELINE_USD = round(
        (PRIOR_MEAN + KAPPA_1 * (THRESHOLD_BASELINE - PRIOR_MEAN)) / POINTS_PER_USD, 2
    )  # = 48/20 = 2.40
    THRESHOLD_WORK_USD = round(
        (PRIOR_MEAN + KAPPA_1 * (THRESHOLD_WORK - PRIOR_MEAN)) / POINTS_PER_USD, 2
    )  # = 52/20 = 2.60
    NOISE_WAGE_RANGE_USD = round(KAPPA_1 * NOISE_Q / POINTS_PER_USD, 2)  # = 4/20 = 0.20


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

    # ── Practice (round 1 only, before Baseline) ─────────────
    # Decision H (H1): unpaid familiarisation round. Performance recorded
    # for the round-covariate learning fit (H2) but does not enter wages,
    # payment, or any wage history.
    practice_num_correct = models.IntegerField(initial=0)
    practice_data_json = models.LongStringField(initial='[]')

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
    # initial=0 so round-1 (calibration) players who skip the Investment
    # page have a well-defined value for downstream computations.
    training_tokens = models.IntegerField(
        label="How many training tokens would you like to purchase?",
        min=0,
        max=C.MAX_TRAINING_TOKENS,
        initial=0,
    )

    # ── Task outcomes ─────────────────────────────────────────
    # Decision B: skill investment shrinks the grid; it does NOT add points.
    # raw_score == task_score == num_correct × SCORE_MULTIPLIER. The
    # `raw_score` alias is retained so downstream code (signal = raw_score +
    # noise) reads the same as in the model writeup.
    num_correct = models.IntegerField(initial=0)
    task_score = models.IntegerField(initial=0)    # num_correct × SCORE_MULTIPLIER
    raw_score = models.IntegerField(initial=0)     # = task_score under Decision B
    grid_rows_used = models.IntegerField(initial=0)  # rows shown this round (after investment)
    grid_cols_used = models.IntegerField(initial=0)  # cols shown this round
    noise_drawn = models.IntegerField(initial=0)   # ε sampled from Uniform[NOISE_MIN, NOISE_MAX]
    signal = models.IntegerField(initial=0)        # raw_score + noise_drawn

    # ── Hiring and wages ──────────────────────────────────────
    hired = models.BooleanField(initial=False)
    wage = models.FloatField(initial=0.0)          # points
    history_premium = models.FloatField(initial=0.0)  # β · Σ(w_j - μ); 0 under SHB

    # ── Financials ────────────────────────────────────────────
    training_cost = models.FloatField(initial=0.0)
    round_net = models.FloatField(initial=0.0)             # wage - training_cost (pre-floor)
    wage_counterfactual = models.FloatField(initial=0.0)          # wage formula value regardless of hiring
    baseline_wage_counterfactual = models.FloatField(initial=0.0)  # same for calibration round

    # ── Task data (serialised JSON for debugging / analysis) ──
    task_data_json = models.LongStringField(initial='[]')

    # ── Survey fields — page 1 (strategy) ────────────────────
    survey_token_factors = models.LongStringField(
        label="What factors did you use in deciding how many tokens to buy? Please give a detailed description of your thought process.",
        blank=True,
    )
    survey_change_over_rounds = models.LongStringField(
        label="How did this change over rounds?",
        blank=True,
    )

    # ── Survey fields — page 1 (demographics) ────────────────
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

    # ── Survey fields — page 2 (experiment feedback) ──────────
    survey_experiment_thoughts = models.LongStringField(
        label="Do you have any thoughts on how this experiment could better accomplish this aim? Please provide any thoughts you have.",
        blank=True,
    )
    survey_wording_strength = models.StringField(
        label="In your opinion, is the amount of/strength of wording in prompting on salary history availability/non-availability in the condition you were assigned to not strong enough, neutral, or too strong?",
        choices=['Not strong enough', 'Neutral', 'Too strong'],
        blank=True,
    )
    survey_wording_explain = models.LongStringField(
        label="Explain your answer.",
        blank=True,
    )
    survey_user_friendly = models.LongStringField(
        label="Is there any way the experiment could be more user-friendly?",
        blank=True,
    )
    survey_problems = models.LongStringField(
        label="Did you experience any problems?",
        blank=True,
    )
    survey_name = models.StringField(
        label="Please provide your name here if you are willing so I can contact you with any questions about your responses.",
        blank=True,
    )

    # ── Comprehension check ───────────────────────────────────
    comp_check_tokens = models.StringField(
        label="What did training tokens do in this study?",
        choices=[
            'They reduced the size of the counting grid',
            'They directly increased my wage offer',
            'They gave me hints during the task',
            'They changed the wage formula',
        ],
        blank=True,
    )
    comp_check_info = models.StringField(
        label="What did the employer use to determine your wage?",
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


def compute_wage(player: Player) -> tuple[float, float]:
    """
    Compute the wage for this round under the player's condition.

    Decision D (D2): wage-history additive form. Section III's wage
    functions are kept general; this is a particular functional form
    satisfying A1–A8 with A6 weak (Channel I = 0) and A3, A4 strict
    (Channels II and III deliver Proposition 1).

      SHB:  w_t = μ + κ₁ (s_t - μ)
      NB :  w_t = μ + κ₁ (s_t - μ) + β · Σ_{j<t} (w_j - μ)

    Past wages are pulled from `baseline_wage_offer` for round 1 (the
    calibration round under A1) and from `wage` for later rounds.

    Returns:
        (wage, history_premium)
    """
    mu = C.PRIOR_MEAN
    base = mu + C.KAPPA_1 * (player.signal - mu)

    if get_condition(player) == 'no_shb':
        past_wages = []
        for p in player.in_previous_rounds():
            w_prev = p.baseline_wage_offer if p.round_number == 1 else p.wage
            past_wages.append(w_prev)
        history_premium = C.HISTORY_BETA * sum(w - mu for w in past_wages)
    else:
        history_premium = 0.0

    wage = base + history_premium
    return (round(wage, 2), round(history_premium, 2))


def get_salary_history(player: Player) -> list[dict]:
    """
    Return a list of dicts with historical round data for display.
    Used in Investment and Results pages under No-SHB condition.

    Under A1, round 1 is the calibration round (t=0): the relevant fields
    live in baseline_*, no investment was made. Rounds 2-3 use the regular
    fields. The display field `is_calibration` lets templates flag this row
    differently if desired.
    """
    history = []
    for pr in player.in_previous_rounds():
        if pr.round_number == 1:
            history.append({
                'round': pr.round_number,
                'tokens': 0,
                'signal': pr.baseline_signal,
                'hired': pr.baseline_hired,
                'wage': round(pr.baseline_wage_offer, 2),
                'wage_usd': f"{pr.baseline_wage_offer / C.POINTS_PER_USD:.2f}",
                'is_calibration': True,
            })
        else:
            history.append({
                'round': pr.round_number,
                'tokens': pr.training_tokens,
                'signal': pr.signal,
                'hired': pr.hired,
                'wage': round(pr.wage, 2),
                'wage_usd': f"{pr.wage / C.POINTS_PER_USD:.2f}",
                'is_calibration': False,
            })
    return history


def format_currency(val: float) -> str:
    return f"{val:.2f}"


def compute_bank_balance(player: Player) -> float:
    """
    Spendable token bank in USD at the start of this investment decision.
    = token allowance + wages earned in all previous rounds - token costs already spent.

    Note: base pay was removed (BASE_PAY_POINTS = 0), so the bank is just the
    fungible $5 allowance (TOKEN_ENDOWMENT_POINTS) plus wages earned so far,
    minus token costs already spent. Always ≥ 0 when the budget constraint is
    enforced.
    """
    balance = C.TOKEN_ENDOWMENT_POINTS
    for pr in player.in_previous_rounds():
        if pr.round_number == 1:
            balance += pr.baseline_wage_offer   # 0 if not hired
        else:
            balance += pr.wage                  # 0 if not hired
            balance -= pr.training_cost
    return round(balance / C.POINTS_PER_USD, 2)


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
            'participation_fee_usd': f"{C.BASE_PAY_POINTS / C.POINTS_PER_USD:.2f}",
            'base_pay_usd': f"{C.BASE_PAY_POINTS / C.POINTS_PER_USD:.2f}",
            'token_endowment_usd': f"{C.TOKEN_ENDOWMENT_POINTS / C.POINTS_PER_USD:.2f}",
            'points_per_usd': C.POINTS_PER_USD,
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
        # Pre-compute example wages for instructions (Decision D, D2).
        # Example: signal s = 65 in current round, prior wage w_0 = 58
        # (corresponds to a calibration signal of s_0 = 70).
        mu = C.PRIOR_MEAN
        example_signal = 65
        example_w0 = 58
        example_w_shb = mu + C.KAPPA_1 * (example_signal - mu)
        example_w_noshb_t1 = (
            mu + C.KAPPA_1 * (example_signal - mu)
            + C.HISTORY_BETA * (example_w0 - mu)
        )
        # Demo grid (5 rows × 6 cols) for the instructions illustration
        demo_grid = [3, 7, 2, 9, 1, 4, 0, 5, 7, 8, 2, 6,
                     1, 4, 7, 3, 6, 0, 9, 2, 1, 4, 7, 5,
                     0, 8, 3, 6, 2, 1]
        demo_cells = [{'digit': d, 'is_target': (d == 7)} for d in demo_grid]

        # Token-by-grid-by-cost menu for the cost/benefit table in the instructions
        token_menu = []
        for t in range(C.MAX_TRAINING_TOKENS + 1):
            rows, cols = C.GRID_BY_TOKENS[t]
            token_menu.append({
                'tokens': t,
                'cost': C.COST_BY_TOKENS[t],
                'cost_usd': f"{C.COST_BY_TOKENS[t] / C.POINTS_PER_USD:.2f}",
                'rows': rows,
                'cols': cols,
                'cells': rows * cols,
            })
        baseline_cells = C.GRID_BY_TOKENS[0][0] * C.GRID_BY_TOKENS[0][1]

        return {
            'condition': condition,
            'is_shb': is_shb,
            'max_tokens': C.MAX_TRAINING_TOKENS,
            'max_token_cost': C.COST_BY_TOKENS[C.MAX_TRAINING_TOKENS],
            'token_cost_1': C.COST_BY_TOKENS[1],
            'token_cost_2': C.COST_BY_TOKENS[2],
            'token_menu': token_menu,
            'baseline_cells': baseline_cells,
            'demo_cells': demo_cells,
            'threshold_baseline': C.THRESHOLD_BASELINE,
            'threshold_work': C.THRESHOLD_WORK,
            'noise_min': C.NOISE_MIN,
            'noise_max': C.NOISE_MAX,
            'num_rounds': C.NUM_ROUNDS,
            'task_duration': C.TASK_DURATION_SECONDS,
            'score_multiplier': C.SCORE_MULTIPLIER,
            'kappa_1': C.KAPPA_1,
            'history_beta': C.HISTORY_BETA,
            'prior_mean': int(C.PRIOR_MEAN),
            'example_signal': example_signal,
            'example_w0': example_w0,
            'example_w_shb': round(example_w_shb, 1),
            'example_w_noshb_t1': round(example_w_noshb_t1, 1),
            'participation_fee_usd': f"{C.BASE_PAY_POINTS / C.POINTS_PER_USD:.2f}",
            'base_pay_usd': f"{C.BASE_PAY_POINTS / C.POINTS_PER_USD:.2f}",
            'token_endowment_usd': f"{C.TOKEN_ENDOWMENT_POINTS / C.POINTS_PER_USD:.2f}",
            'points_per_usd': C.POINTS_PER_USD,
            'calibration_duration': C.BASELINE_DURATION_SECONDS,
            'threshold_baseline_usd': f"{C.THRESHOLD_BASELINE_USD:.2f}",
            'threshold_work_usd': f"{C.THRESHOLD_WORK_USD:.2f}",
            'prior_mean_usd': f"{C.PRIOR_MEAN / C.POINTS_PER_USD:.2f}",
            'noise_wage_range_usd': f"{C.NOISE_WAGE_RANGE_USD:.2f}",
            'example_w_shb_usd': f"{round(example_w_shb, 2) / C.POINTS_PER_USD:.2f}",
            'example_w_noshb_t1_usd': f"{round(example_w_noshb_t1, 2) / C.POINTS_PER_USD:.2f}",
            'example_w0_usd': f"{example_w0 / C.POINTS_PER_USD:.2f}",
        }


class Practice(Page):
    """
    Unpaid practice round (round 1 only, before Baseline).

    Decision H (H1, 2026-04-27): workers complete a brief familiarisation
    round so the calibration round is not the first time they see the
    counting task. Performance is recorded (`practice_num_correct`,
    `practice_data_json`) so that the analysis can fit a round-by-round
    learning curve (Decision H2: round-number covariate). It does NOT
    affect any wage offer, the calibration signal, or final payment.
    """

    form_model = 'player'
    form_fields = ['practice_num_correct', 'practice_data_json']

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def get_timeout_seconds(player: Player):
        return C.PRACTICE_DURATION_SECONDS

    @staticmethod
    def vars_for_template(player: Player):
        rows, cols = C.GRID_BY_TOKENS[0]
        expiry = getattr(player.participant, '_timeout_expiration_time', None)
        server_remaining = max(0, round(expiry - time.time())) if expiry else C.PRACTICE_DURATION_SECONDS
        return {
            'task_duration': C.PRACTICE_DURATION_SECONDS,
            'grid_rows': rows,
            'grid_cols': cols,
            'server_remaining': server_remaining,
        }


class GetReady(Page):
    """
    Short transition page shown after Practice and before Baseline (round 1
    only). Lets the participant pause and read what comes next, since the
    Practice page auto-submits when its timer expires.
    """

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player: Player):
        return {
            'baseline_duration': C.BASELINE_DURATION_SECONDS,
            'threshold_baseline': C.THRESHOLD_BASELINE,
            'threshold_baseline_usd': f"{C.THRESHOLD_BASELINE_USD:.2f}",
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
        expiry = getattr(player.participant, '_timeout_expiration_time', None)
        server_remaining = max(0, round(expiry - time.time())) if expiry else C.BASELINE_DURATION_SECONDS
        return {
            'task_duration': C.BASELINE_DURATION_SECONDS,
            'grid_sizes_json': json.dumps(C.BASELINE_GRID_SIZES),
            'server_remaining': server_remaining,
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        # Compute private baseline signal
        player.baseline_score = player.baseline_num_correct * C.SCORE_MULTIPLIER
        player.baseline_noise = random.randint(C.NOISE_MIN, C.NOISE_MAX)
        player.baseline_signal = player.baseline_score + player.baseline_noise

        # Calibration wage: single-signal formula w_0 = μ + κ₁(s_0 - μ),
        # identical under both regimes since no history exists at t=0.
        # Retention threshold r_0 is lower than the work-round threshold (Decision F).
        # Always compute the formula wage (for counterfactual display when not hired)
        formula_wage = round(
            C.PRIOR_MEAN + C.KAPPA_1 * (player.baseline_signal - C.PRIOR_MEAN), 2
        )
        player.baseline_wage_counterfactual = formula_wage
        if player.baseline_signal >= C.THRESHOLD_BASELINE:
            player.baseline_hired = True
            player.baseline_wage_offer = formula_wage
        else:
            player.baseline_hired = False
            player.baseline_wage_offer = 0.0

        # Persist on participant for retrieval in later rounds (e.g. for
        # No-SHB wage history if you decide to include the baseline signal)
        player.participant.vars['baseline_signal'] = player.baseline_signal
        player.participant.vars['baseline_wage_offer'] = player.baseline_wage_offer
        player.participant.vars['baseline_hired'] = player.baseline_hired

        # A1: round-1 wage w_0 is paid (no training cost in calibration).
        player.payoff = player.baseline_wage_offer


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
            'wage_offer_usd': f"{player.baseline_wage_offer / C.POINTS_PER_USD:.2f}",
            'wage_counterfactual_usd': f"{player.baseline_wage_counterfactual / C.POINTS_PER_USD:.2f}",
            'threshold_usd': f"{C.THRESHOLD_BASELINE_USD:.2f}",
            'hired': player.baseline_hired,
            'is_shb': (get_condition(player) == 'shb'),
            'threshold': C.THRESHOLD_BASELINE,
            'prior_mean': int(C.PRIOR_MEAN),
        }


class Investment(Page):
    """
    Investment decision stage.
    Participant chooses training_tokens (0–3).
    No-SHB participants see their salary history.

    A1: skipped in round 1 (calibration / t=0). First investment is in round 2 (t=1).
    """

    form_model = 'player'
    form_fields = ['training_tokens']

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number > 1

    @staticmethod
    def error_message(player: Player, values: dict):
        tokens = values.get('training_tokens', 0)
        bank_usd = compute_bank_balance(player)
        cost_usd = C.COST_BY_TOKENS[tokens] / C.POINTS_PER_USD
        if cost_usd > bank_usd:
            return (
                f"You cannot afford {tokens} token{'s' if tokens != 1 else ''}. "
                f"Cost: ${cost_usd:.2f}, your available balance: ${bank_usd:.2f}."
            )

    @staticmethod
    def vars_for_template(player: Player):
        condition = get_condition(player)
        is_shb = (condition == 'shb')
        history = get_salary_history(player)
        bank_balance_usd = compute_bank_balance(player)

        # Show cost–grid menu (Decision B: tokens shrink the grid;
        # Decision C: cumulative cost rises 0 → 10 → 24 → 48 points).
        # cost_usd is the dollar equivalent shown to participants.
        # affordable flags whether the participant's current bank covers this option.
        token_table = []
        for t in range(C.MAX_TRAINING_TOKENS + 1):
            rows, cols = C.GRID_BY_TOKENS[t]
            cost_points = C.COST_BY_TOKENS[t]
            cost_usd = cost_points / C.POINTS_PER_USD
            token_table.append({
                'tokens': t,
                'cost': cost_points,
                'cost_usd': f"{cost_usd:.2f}",
                'rows': rows,
                'cols': cols,
                'cells': rows * cols,
                'affordable': cost_usd <= bank_balance_usd,
            })

        return {
            'round_number': player.round_number,
            'work_round': player.round_number - 1,
            'num_rounds': C.NUM_ROUNDS,
            'num_work_rounds': C.NUM_ROUNDS - 1,
            'is_shb': is_shb,
            'condition': condition,
            'history': history,
            'history_count_plus_1': len(history) + 1,
            'token_table': token_table,
            'max_tokens': C.MAX_TRAINING_TOKENS,
            'threshold': C.THRESHOLD_WORK,
            'noise_min': C.NOISE_MIN,
            'noise_max': C.NOISE_MAX,
            'task_duration': C.TASK_DURATION_SECONDS,
            'kappa_1': C.KAPPA_1,
            'history_beta': C.HISTORY_BETA,
            'prior_mean': int(C.PRIOR_MEAN),
            'has_history': (len(history) > 0),
            'threshold_usd': f"{C.THRESHOLD_WORK_USD:.2f}",
            'bank_balance_usd': f"{bank_balance_usd:.2f}",
        }


class TaskReady(Page):
    """
    Brief transition between Investment (where the token decision is made)
    and Task (the timed counting round). Shows a summary of what the
    participant just chose and warns them the timer starts immediately on
    click. Round 2-3 only (work rounds).

    Added 2026-05-03: previously the timer started the instant the
    participant clicked Next on Investment, with no chance to pause and
    actually be ready. This page closes that gap.
    """

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number > 1

    @staticmethod
    def vars_for_template(player: Player):
        rows, cols = C.GRID_BY_TOKENS[player.training_tokens]
        cost_points = C.COST_BY_TOKENS[player.training_tokens]
        return {
            'round_number': player.round_number,
            'work_round': player.round_number - 1,
            'num_work_rounds': C.NUM_ROUNDS - 1,
            'training_tokens': player.training_tokens,
            'training_cost': cost_points,
            'training_cost_usd': f"{cost_points / C.POINTS_PER_USD:.2f}",
            'grid_rows': rows,
            'grid_cols': cols,
            'grid_cells': rows * cols,
            'task_duration': C.TASK_DURATION_SECONDS,
            'threshold': C.THRESHOLD_WORK,
            'threshold_usd': f"{C.THRESHOLD_WORK_USD:.2f}",
        }


class Task(Page):
    """
    Real-effort number-counting task (2 minutes).
    Grid and problem generation handled in JavaScript (Task.html).
    num_correct is written to a hidden field by JS on timeout/submit.

    A1: skipped in round 1. The calibration task lives on the Baseline page.
    """

    form_model = 'player'
    form_fields = ['num_correct', 'task_data_json']

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number > 1

    @staticmethod
    def get_timeout_seconds(player: Player):
        return C.TASK_DURATION_SECONDS

    @staticmethod
    def vars_for_template(player: Player):
        # Decision B: grid dimensions are determined by tokens purchased.
        rows, cols = C.GRID_BY_TOKENS[player.training_tokens]
        # Persist for downstream display / analysis (the actual write happens
        # in before_next_page; we mirror it here so the template renders).
        expiry = getattr(player.participant, '_timeout_expiration_time', None)
        server_remaining = max(0, round(expiry - time.time())) if expiry else C.TASK_DURATION_SECONDS
        return {
            'round_number': player.round_number,
            'work_round': player.round_number - 1,
            'num_rounds': C.NUM_ROUNDS,
            'num_work_rounds': C.NUM_ROUNDS - 1,
            'training_tokens': player.training_tokens,
            'task_duration': C.TASK_DURATION_SECONDS,
            'threshold': C.THRESHOLD_WORK,
            'noise_min': C.NOISE_MIN,
            'noise_max': C.NOISE_MAX,
            'grid_rows': rows,
            'grid_cols': cols,
            'grid_cells': rows * cols,
            'score_multiplier': C.SCORE_MULTIPLIER,
            'server_remaining': server_remaining,
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        """
        After task: compute scores, draw noise, set signal, compute wage.
        """
        # 0. Record the grid actually shown this round (Decision B).
        rows, cols = C.GRID_BY_TOKENS[player.training_tokens]
        player.grid_rows_used = rows
        player.grid_cols_used = cols

        # 1. Compute scores. Under Decision B, raw_score == task_score; the
        # skill investment paid for an easier task, not for bonus points.
        player.task_score = player.num_correct * C.SCORE_MULTIPLIER
        player.raw_score = player.task_score

        # 2. Draw noise and compute signal
        player.noise_drawn = random.randint(C.NOISE_MIN, C.NOISE_MAX)
        player.signal = player.raw_score + player.noise_drawn

        # 3. Determine hiring (work-round threshold r_1 = r_2; Decision F)
        player.hired = (player.signal >= C.THRESHOLD_WORK)

        # 4. Compute wage formula always (hired or not) for counterfactual display;
        # apply to player.wage only if hired (Decision D, D2).
        wage_formula, premium_formula = compute_wage(player)
        player.wage_counterfactual = max(0.0, round(wage_formula, 2))
        if player.hired:
            player.wage = player.wage_counterfactual
            player.history_premium = premium_formula
        else:
            player.wage = 0.0
            player.history_premium = 0.0

        # 5. Compute net earnings this round (Decision C: cumulative cost is convex)
        player.training_cost = float(C.COST_BY_TOKENS[player.training_tokens])
        player.round_net = player.wage - player.training_cost

        # 6. Set oTree payoff = full round net (NOT floored at 0). The token
        # cost is borne in full out of the fungible allowance + earned wages,
        # so over-investing is a real loss (Harrison). A round can be negative
        # (e.g. bought a token but was not hired). With base pay removed there
        # is NO guaranteed floor: the only fixed component added at the end is
        # the $5 fungible allowance, so a participant who spends it on tokens
        # and is not hired can finish near $0. The budget constraint
        # (compute_bank_balance) still blocks purchases the bank can't cover.
        player.payoff = player.round_net


class Results(Page):
    """
    Round results page.
    Shows score breakdown, signal, hiring outcome, wage, and history (No-SHB only).

    A1: skipped in round 1. Calibration outcome is shown on BaselineResult.
    """

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number > 1

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
            'wage_usd': f"{player.wage / C.POINTS_PER_USD:.2f}",
        }]

        # Wage decomposition for display (Decision D, D2):
        #   wage = [μ + κ₁(s - μ)]  +  history_premium
        # history_premium is 0 under SHB (and 0 in any round if not hired).
        if player.hired:
            wage_from_current = C.PRIOR_MEAN + C.KAPPA_1 * (player.signal - C.PRIOR_MEAN)
            history_premium = player.history_premium
        else:
            wage_from_current = player.wage
            history_premium = 0.0

        return {
            'round_number': player.round_number,
            'work_round': player.round_number - 1,
            'num_rounds': C.NUM_ROUNDS,
            'num_work_rounds': C.NUM_ROUNDS - 1,
            'is_shb': is_shb,
            'condition': condition,
            'training_tokens': player.training_tokens,
            'training_cost': player.training_cost,
            'grid_rows_used': player.grid_rows_used,
            'grid_cols_used': player.grid_cols_used,
            'grid_cells_used': player.grid_rows_used * player.grid_cols_used,
            'num_correct': player.num_correct,
            'task_score': player.task_score,
            'raw_score': player.raw_score,
            'noise_drawn': player.noise_drawn,
            'signal': player.signal,
            'hired': player.hired,
            'wage': round(player.wage, 2),
            'wage_usd': f"{player.wage / C.POINTS_PER_USD:.2f}",
            'wage_counterfactual_usd': f"{player.wage_counterfactual / C.POINTS_PER_USD:.2f}",
            'training_cost_usd': f"{player.training_cost / C.POINTS_PER_USD:.2f}",
            'net_floored_usd': f"{max(0, player.round_net) / C.POINTS_PER_USD:.2f}",
            'round_net_usd': f"{player.round_net / C.POINTS_PER_USD:.2f}",
            'threshold_usd': f"{C.THRESHOLD_WORK_USD:.2f}",
            'kappa_1': C.KAPPA_1,
            'history_beta': C.HISTORY_BETA,
            'round_net': round(player.round_net, 2),
            'net_floored': round(max(0, player.round_net), 2),
            'threshold': C.THRESHOLD_WORK,
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

        # A1 helper: per-round row, pulling from baseline_* in round 1
        # (calibration / t=0) and from regular fields in rounds 2-3 (t=1, t=2).
        history = []
        for r in all_rounds:
            if r.round_number == 1:
                history.append({
                    'round': r.round_number,
                    'tokens': 0,
                    'cost': 0.0,
                    'cost_usd': "0.00",
                    'grid': '×'.join(str(x) for x in C.GRID_BY_TOKENS[0]),
                    'correct': r.baseline_num_correct,
                    'task_score': r.baseline_score,
                    'raw_score': r.baseline_score,
                    'noise': r.baseline_noise,
                    'signal': r.baseline_signal,
                    'hired': r.baseline_hired,
                    'wage': round(r.baseline_wage_offer, 2),
                    'wage_usd': f"{r.baseline_wage_offer / C.POINTS_PER_USD:.2f}",
                    'net': round(r.baseline_wage_offer, 2),
                    'net_usd': f"{r.baseline_wage_offer / C.POINTS_PER_USD:.2f}",
                    'is_calibration': True,
                })
            else:
                grid_str = f"{r.grid_rows_used}×{r.grid_cols_used}" if r.grid_rows_used else '—'
                history.append({
                    'round': r.round_number,
                    'tokens': r.training_tokens,
                    'cost': round(r.training_cost, 2),
                    'cost_usd': f"{r.training_cost / C.POINTS_PER_USD:.2f}",
                    'grid': grid_str,
                    'correct': r.num_correct,
                    'task_score': r.task_score,
                    'raw_score': r.raw_score,
                    'noise': r.noise_drawn,
                    'signal': r.signal,
                    'hired': r.hired,
                    'wage': round(r.wage, 2),
                    'wage_usd': f"{r.wage / C.POINTS_PER_USD:.2f}",
                    'net': round(r.round_net, 2),
                    'net_usd': f"{r.round_net / C.POINTS_PER_USD:.2f}",
                    'is_calibration': False,
                })

        # Totals across all rounds (calibration wage counted; calibration has
        # no training tokens or cost).
        total_wages = sum(
            (r.baseline_wage_offer if r.round_number == 1 else r.wage)
            for r in all_rounds
        )
        total_costs = sum(r.training_cost for r in all_rounds)
        total_correct = sum(
            (r.baseline_num_correct if r.round_number == 1 else r.num_correct)
            for r in all_rounds
        )
        total_tokens = sum(r.training_tokens for r in all_rounds)
        times_hired = sum(
            1 for r in all_rounds
            if (r.baseline_hired if r.round_number == 1 else r.hired)
        )

        # Net game earnings: calibration pays w_0 in full; work rounds pay
        # w_t - training_cost in FULL (token cost borne out of allowance +
        # wages, not floored per round). All amounts in points.
        total_bonus_points = sum(
            (r.baseline_wage_offer if r.round_number == 1 else r.round_net)
            for r in all_rounds
        )
        # Take-home = $5 fungible token allowance + net game earnings (no base).
        total_points = (
            total_bonus_points + C.BASE_PAY_POINTS + C.TOKEN_ENDOWMENT_POINTS
        )
        total_usd = total_points / C.POINTS_PER_USD

        return {
            'history': history,
            'total_wages': round(total_wages, 2),
            'total_costs': round(total_costs, 2),
            'total_correct': total_correct,
            'total_tokens': total_tokens,
            'times_hired': times_hired,
            'total_bonus_points': round(total_bonus_points, 2),
            'participation_fee_points': C.BASE_PAY_POINTS,
            'token_endowment_points': C.TOKEN_ENDOWMENT_POINTS,
            'total_points': round(total_points, 2),
            'total_usd': round(total_usd, 2),
            'bonus_usd': round(total_bonus_points / C.POINTS_PER_USD, 2),
            'participation_usd': C.BASE_PAY_POINTS / C.POINTS_PER_USD,
            'base_pay_usd': f"{C.BASE_PAY_POINTS / C.POINTS_PER_USD:.2f}",
            'token_endowment_usd': f"{C.TOKEN_ENDOWMENT_POINTS / C.POINTS_PER_USD:.2f}",
            'points_per_usd': C.POINTS_PER_USD,
            'condition': get_condition(player),
            'is_shb': (get_condition(player) == 'shb'),
            'num_rounds': C.NUM_ROUNDS,
        }


class Survey(Page):
    """
    Post-task survey page 1: comprehension checks, token strategy, demographics.
    Payoff is finalised here so it is set before FinalResults / Exit display it.
    Back-navigation is blocked via history.pushState (same as the timed task pages).
    """

    form_model = 'player'
    form_fields = [
        'comp_check_tokens',
        'comp_check_info',
        'survey_token_factors',
        'survey_change_over_rounds',
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
            'correct_comp_answer': 'They reduced the size of the counting grid',
            'correct_info_answer': (
                'Only your performance in this round'
                if is_shb else
                'Your performance in this round AND your full salary history'
            ),
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        """
        Finalise payoff: add the $5 fungible token allowance (base pay = 0).
        Must run here (page 1 of survey) so the total is correct by the time
        FinalResults and Exit render it.
        """
        if player.round_number == C.NUM_ROUNDS:
            player.participant.payoff += (
                C.BASE_PAY_POINTS + C.TOKEN_ENDOWMENT_POINTS
            )


class SurveyFeedback(Page):
    """
    Post-task survey page 2: experiment debrief + qualitative feedback.
    Shown only on the last round, immediately after Survey.
    Back-navigation is blocked via history.pushState.
    """

    form_model = 'player'
    form_fields = [
        'survey_experiment_thoughts',
        'survey_wording_strength',
        'survey_wording_explain',
        'survey_user_friendly',
        'survey_problems',
        'survey_name',
    ]

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == C.NUM_ROUNDS


class Exit(Page):
    """
    Final goodbye page (last round only). Confirms the study is complete,
    repeats the total payment, and surfaces the Prolific completion link if
    one is configured. Without this, oTree drops the participant on its
    default "out of pages" screen with no closure.
    """

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == C.NUM_ROUNDS

    @staticmethod
    def vars_for_template(player: Player):
        all_rounds = player.in_all_rounds()
        total_bonus_points = sum(
            (r.baseline_wage_offer if r.round_number == 1 else r.round_net)
            for r in all_rounds
        )
        total_points = (
            total_bonus_points + C.BASE_PAY_POINTS + C.TOKEN_ENDOWMENT_POINTS
        )
        total_usd = total_points / C.POINTS_PER_USD
        completion_url = (player.session.config.get('prolific_completion_url') or '').strip()
        return {
            'total_bonus_points': round(total_bonus_points, 2),
            'total_usd': round(total_usd, 2),
            'bonus_usd': round(total_bonus_points / C.POINTS_PER_USD, 2),
            'participation_usd': f"{C.BASE_PAY_POINTS / C.POINTS_PER_USD:.2f}",
            'base_pay_usd': f"{C.BASE_PAY_POINTS / C.POINTS_PER_USD:.2f}",
            'token_endowment_usd': f"{C.TOKEN_ENDOWMENT_POINTS / C.POINTS_PER_USD:.2f}",
            'completion_url': completion_url,
            'has_completion_url': bool(completion_url),
        }


# ─────────────────────────────────────────────────────────────
#  PAGE SEQUENCE
# ─────────────────────────────────────────────────────────────

page_sequence = [
    Consent,
    Instructions,
    Practice,         # round 1 only — unpaid familiarisation (Decision H1)
    GetReady,         # round 1 only — pause between Practice and Baseline
    Baseline,         # round 1 only — calibration
    BaselineResult,   # round 1 only — show wage offer (private signal)
    Investment,
    TaskReady,        # round 2-3 only — pause before the timed task starts
    Task,
    Results,
    FinalResults,
    Survey,
    SurveyFeedback,   # last round only — experiment debrief & qualitative feedback
    Exit,             # last round only — final goodbye / Prolific link
]
