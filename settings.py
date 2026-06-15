"""
settings.py — oTree 5 configuration for the SHB Experiment
"""

from os import environ

# Heroku config vars are case-sensitive. Normalize the lowercase names that
# are easy to enter from the dashboard so oTree and this settings file see the
# uppercase names they expect.
for _key in [
    'DATABASE_URL',
    'OTREE_ADMIN_PASSWORD',
    'OTREE_PRODUCTION',
    'OTREE_SECRET_KEY',
]:
    _lower_key = _key.lower()
    if not environ.get(_key) and environ.get(_lower_key):
        environ[_key] = environ[_lower_key]

SESSION_CONFIGS = [
    # ── Working app (post-A1..H revision) ─────────────────────
    {
        'name': 'shb_experiment',
        'display_name': "SHB Experiment — Working (1 calibration + 2 work rounds)",
        'app_sequence': ['shb_experiment'],
        'num_demo_participants': 4,
        'use_browser_bots': False,
        # 2026-06-03: must equal 1 / C.POINTS_PER_USD (=1/35) so the dollars
        # oTree actually pays match the dollar amounts the app shows participants.
        'real_world_currency_per_point': 1 / 20,  # = $0.05 per point
        # Must stay 0.00. Base pay was removed (2026-06-10); the app pays the
        # $5 fungible token allowance itself, in points (Survey.before_next_page
        # adds C.TOKEN_ENDOWMENT_POINTS to payoff, and FinalResults/Exit show a
        # total that includes it). A nonzero oTree participation_fee here would
        # be ADDED on top a second time, double-paying and making the shown
        # total wrong. All pay lives entirely in the app.
        'participation_fee': 0.00,
        'prolific_completion_url': '',
    },
    # # ── Full experiment (production / Prolific) ───────────────
    # {
    #     'name': 'shb_experiment_v0_baseline_full',
    #     'display_name': "SHB Experiment v0 Baseline — Full (3 rounds, real payment)",
    #     'app_sequence': ['shb_experiment_v0_baseline'],
    #     'num_demo_participants': 4,
    #     'use_browser_bots': False,
    #     # Custom session-level configs (accessible via session.config[...])
    #     'real_world_currency_per_point': 0.10,  # $0.10 per point
    #     'participation_fee': 3.00,              # $3.00 base pay (handled in app)
    #     'prolific_completion_url': '',           # Set to your Prolific completion URL
    # },
    # # ── Short demo / pilot (1 round, no real payment) ─────────
    # {
    #     'name': 'shb_experiment_v0_baseline_demo',
    #     'display_name': "SHB Experiment v0 Baseline — Demo (1 round, for piloting)",
    #     'app_sequence': ['shb_experiment_v0_baseline'],
    #     'num_demo_participants': 4,
    #     'use_browser_bots': False,
    #     'real_world_currency_per_point': 0.0,
    #     'participation_fee': 0.0,
    # },
]

# ── oTree configuration ────────────────────────────────────────
SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1 / 20,  # = $0.05; keep in sync with C.POINTS_PER_USD
    participation_fee=0.00,                 # base pay is handled in-app (in points); see note above
    doc="",
)

PARTICIPANT_FIELDS = [
    'condition',         # 'shb' or 'no_shb' — assigned once in round 1
    'prolific_id',       # captured via URL parameter from Prolific
]

SESSION_FIELDS = []

# ── Internationalisation ───────────────────────────────────────
LANGUAGE_CODE = 'en'
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = False        # We use ECU (stored as floats); convert to USD in app

# ── Admin ──────────────────────────────────────────────────────
ADMIN_USERNAME = 'admin'
# Use environment variable for password in production
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD', 'change_me_in_production')

DEMO_PAGE_INTRO_HTML = """
<p>This is a demo of the <strong>Salary History Ban Experiment</strong>.</p>
<p>Participants are randomly assigned to either the <em>SHB condition</em>
(employer cannot see salary history) or the <em>No-SHB condition</em>
(employer can see full salary history).</p>
<p>The experiment tests whether salary history availability affects
human capital investment incentives.</p>
"""

SECRET_KEY = environ.get('OTREE_SECRET_KEY', '{{ secret_key }}')

# ── Debug mode ─────────────────────────────────────────────────
# Set OTREE_PRODUCTION=1 in environment to disable debug mode in production
DEBUG = not environ.get('OTREE_PRODUCTION')

# ── Prolific URL parameter (capture participant ID) ────────────
# In Prolific study settings, set completion URL redirect and pass
# the PROLIFIC_PID as a URL parameter:
#   https://your-server.com/InitializeParticipant?participant_label={{%PROLIFIC_PID%}}

# ── Room for Prolific (optional) ──────────────────────────────
ROOMS = [
    # {
    #     'name': 'prolific_shb',
    #     'display_name': 'SHB Experiment (Prolific)',
    #     'participant_label_file': '_rooms/prolific_shb.txt',  # optional pre-generated labels
    #     'use_secure_urls': True,
    # },
    dict(
        name='econ_lab',
        display_name='Experimental Economics Lab'
    ),

]
