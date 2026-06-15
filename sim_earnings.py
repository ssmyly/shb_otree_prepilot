"""
sim_earnings.py  —  Monte Carlo estimate of participant take-home pay.

Replicates the EXACT payoff logic of shb_experiment/__init__.py
(wages, hiring thresholds, No-SHB wage-history premium, training costs,
budget constraint, and the base + allowance + net-wages payout) and drives
it with a model of human performance on the real-effort counting task.

The one thing the app cannot know in advance is how many grids a real human
counts in 90 s. That is modeled here from a per-person counting speed that is
RIGHT-SKEWED: most people are relatively fast, a minority are slow.

All economic constants below are copied from class C in __init__.py.
"""

import random
import statistics as st

# ───────────────────────── app constants (from class C) ─────────────────────
SCORE_MULT      = 5
NOISE_LO, NOISE_HI = -10, 10          # epsilon ~ discrete uniform
MU              = 50.0                 # prior mean
KAPPA1          = 0.40
BETA            = 0.5                  # No-SHB history weight
THRESH_BASE     = 45                   # calibration hire threshold (signal)
THRESH_WORK     = 55                   # work-round hire threshold (signal)
COST  = {0: 0, 1: 10, 2: 24, 3: 48}            # training cost, points
CELLS = {0: 30, 1: 25, 2: 20, 3: 16}           # grid cells by tokens (work rounds)
BASELINE_SIZES  = [12, 20, 30]         # calibration cycles these (3x4,4x5,5x6)
BASE_PAY        = 60                    # $3.00 guaranteed, points
TOKEN_ENDOW     = 40                    # $2.00 fungible allowance, points
PPU             = 20                    # points per USD
DURATION        = 90                    # seconds per round
MAX_TOKENS      = 3

# ───────────────────────── performance model (assumptions) ──────────────────
# time to dispatch one grid = per_cell * cells + OVERHEAD
#   OVERHEAD = read target + type answer + click  + 0.7s forced feedback gap
OVERHEAD        = 2.3
# per-person counting speed (sec/cell), right-skewed via lognormal.
# base case median 0.26 s/cell; slow tail reaches ~0.7-0.9 s/cell.
ACC_MEAN, ACC_SD = 0.90, 0.05          # per-person base accuracy
ACC_GRID_PENALTY = 0.0015              # accuracy drop per cell above 16
ROUND_JITTER_SD  = 0.12                # round-to-round speed wobble (lognormal)


def draw_person(speed_median, speed_sdlog):
    per_cell = speed_median * random.lognormvariate(0.0, speed_sdlog)
    per_cell = min(max(per_cell, 0.12), 2.0)
    acc = min(max(random.gauss(ACC_MEAN, ACC_SD), 0.55), 0.99)
    return per_cell, acc


def play_round(per_cell_person, acc, sizes):
    """Simulate one 90s round. `sizes` is an int (fixed grid) or a list."""
    per_cell = per_cell_person * random.lognormvariate(0.0, ROUND_JITTER_SD)
    t = 0.0
    nc = 0
    while True:
        cells = sizes if isinstance(sizes, int) else random.choice(sizes)
        ptime = per_cell * cells + OVERHEAD
        if t + ptime > DURATION:        # ran out of time mid-problem -> not counted
            break
        t += ptime
        a = max(0.50, acc - ACC_GRID_PENALTY * (cells - 16))
        if random.random() < a:
            nc += 1
    return nc


def wage_given_signal(signal, history_premium):
    base = MU + KAPPA1 * (signal - MU)
    return max(0.0, round(base + history_premium, 2))


def expected_net(score_hat, cost, premium):
    """Expected round net over the 21 noise outcomes (used by rational chooser)."""
    tot = 0.0
    for n in range(NOISE_LO, NOISE_HI + 1):
        sig = score_hat + n
        if sig >= THRESH_WORK:
            tot += wage_given_signal(sig, premium)
    ev_wage = tot / (NOISE_HI - NOISE_LO + 1)
    return ev_wage - cost


def choose_tokens_rational(per_cell, acc, bank_pts, premium):
    best_t, best_ev = 0, -1e9
    for t in range(MAX_TOKENS + 1):
        if COST[t] > bank_pts:           # budget constraint (app blocks this)
            continue
        cells = CELLS[t]
        nc_hat = round(acc * DURATION / (per_cell * cells + OVERHEAD))
        ev = expected_net(nc_hat * SCORE_MULT, COST[t], premium)
        if ev > best_ev:
            best_ev, best_t = ev, t
    return best_t


NAIVE_DIST = {0: 0.20, 1: 0.30, 2: 0.30, 3: 0.20}

def choose_tokens_naive(bank_pts):
    r, c = random.random(), 0.0
    for t in range(MAX_TOKENS + 1):
        c += NAIVE_DIST[t]
        if r <= c and COST[t] <= bank_pts:
            return t
    # fall back to the most expensive affordable token at/below the draw
    affordable = [t for t in range(MAX_TOKENS + 1) if COST[t] <= bank_pts]
    return max(affordable)


def simulate_participant(condition, policy, speed_median, speed_sdlog):
    per_cell, acc = draw_person(speed_median, speed_sdlog)
    payoff = 0.0
    past_wages = []                      # for No-SHB premium (incl. calibration)

    # ── Round 1: calibration (varying grid, no investment) ──
    nc0 = play_round(per_cell, acc, BASELINE_SIZES)
    sig0 = nc0 * SCORE_MULT + random.randint(NOISE_LO, NOISE_HI)
    if sig0 >= THRESH_BASE:
        w0 = round(MU + KAPPA1 * (sig0 - MU), 2)
    else:
        w0 = 0.0
    payoff += w0
    past_wages.append(w0)

    # ── Rounds 2-3: work rounds ──
    for _ in range(2):
        # spendable bank = fungible allowance + wages earned so far - costs spent.
        # `payoff` currently holds exactly (w0 + sum work nets so far) = earned - spent.
        # The $3 base is NOT spendable, so it is excluded here.
        bank_pts = TOKEN_ENDOW + payoff
        if condition == 'no_shb':
            premium = BETA * sum(w - MU for w in past_wages)
        else:
            premium = 0.0

        if policy == 'rational':
            t = choose_tokens_rational(per_cell, acc, bank_pts, premium)
        else:
            t = choose_tokens_naive(bank_pts)

        nc = play_round(per_cell, acc, CELLS[t])
        sig = nc * SCORE_MULT + random.randint(NOISE_LO, NOISE_HI)
        hired = sig >= THRESH_WORK
        wage = wage_given_signal(sig, premium) if hired else 0.0
        cost = COST[t]
        payoff += wage - cost
        past_wages.append(wage)

    payoff += BASE_PAY + TOKEN_ENDOW     # add guaranteed base + fungible allowance
    return payoff / PPU                  # take-home USD


def run(policy, speed_median, speed_sdlog=0.40, n=40000):
    out = {'shb': [], 'no_shb': []}
    for cond in ('shb', 'no_shb'):
        for _ in range(n):
            out[cond].append(simulate_participant(cond, policy, speed_median, speed_sdlog))
    allv = out['shb'] + out['no_shb']
    return out, allv


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100 * len(v)))]


def summarize(label, allv, out):
    print(f"\n=== {label} ===")
    print(f"  mean ${st.mean(allv):.2f}   median ${st.median(allv):.2f}   "
          f"sd ${st.pstdev(allv):.2f}")
    print(f"  p10 ${pct(allv,10):.2f}  p25 ${pct(allv,25):.2f}  "
          f"p75 ${pct(allv,75):.2f}  p90 ${pct(allv,90):.2f}  "
          f"min ${min(allv):.2f}  max ${max(allv):.2f}")
    print(f"  mean SHB ${st.mean(out['shb']):.2f}   "
          f"mean No-SHB ${st.mean(out['no_shb']):.2f}")
    floor = sum(1 for x in allv if x <= 5.05) / len(allv)
    print(f"  share at/near $5 floor (base+allowance, no net wages): {floor*100:.1f}%")


if __name__ == '__main__':
    random.seed(12345)
    print("Monte Carlo of take-home pay (USD). N=40,000 per condition per cell.")

    print("\n############ BASE CASE (median 0.26 s/cell) ############")
    for policy in ('rational', 'naive'):
        out, allv = run(policy, 0.26)
        summarize(f"token policy = {policy}", allv, out)

    print("\n\n############ SENSITIVITY: typical counting speed ############")
    print("(rational token choice; slower median => fewer correct => fewer hires)")
    for sm in (0.20, 0.26, 0.32, 0.40):
        out, allv = run('rational', sm, n=20000)
        print(f"\n  median {sm:.2f}s/cell -> mean ${st.mean(allv):.2f}  "
              f"median ${st.median(allv):.2f}  "
              f"SHB ${st.mean(out['shb']):.2f}  NoSHB ${st.mean(out['no_shb']):.2f}")
