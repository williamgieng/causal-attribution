import math
from itertools import permutations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)   # seeded for reproducibility

# Figure styling
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#CCCCCC',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.color': '#EEEEEE',
})

BLUE, GREEN, RED   = '#2563EB', '#16A34A', '#DC2626'
GRAY, DARK, ORANGE = '#9CA3AF', '#1F2937', '#D97706'

N = 10_000   # number of customers in the simulation

# True effects baked into the data, what each method should recover.
TRUE_BASELINE    = 0.08   # 8% baseline 6-month churn (neither force active)
TRUE_PROMO       = 0.05   # +5 pp from promo expiry alone
TRUE_INITIATIVE  = 0.04   # +4 pp from initiative completion alone
TRUE_INTERACTION = 0.05   # +5 pp additional lift when BOTH forces co-occur

customers = pd.DataFrame({
    'customer_id':         np.arange(N),
    'promo_expired':       RNG.choice([0, 1], N, p=[0.45, 0.55]),
    'initiative_complete': RNG.choice([0, 1], N, p=[0.50, 0.50]),
    'arr_usd':             RNG.lognormal(10.5, 0.8, N),   # annual recurring revenue
    'tenure_months':       RNG.uniform(10, 14, N),
    'n_seats':             RNG.integers(5, 200, N),       # licensed seats
})

# Each customer's true churn probability = baseline + promo + init + interaction.
# The interaction term only fires when BOTH forces are active.
churn_prob = (
    TRUE_BASELINE
    + TRUE_PROMO       * customers['promo_expired']
    + TRUE_INITIATIVE  * customers['initiative_complete']
    + TRUE_INTERACTION * customers['promo_expired']
                       * customers['initiative_complete']
)
customers['churned'] = (RNG.uniform(size=N) < churn_prob).astype(int)

print('Cohort sizes:')
print(customers.groupby(['promo_expired', 'initiative_complete']).size())

N_WEEKS_PRE   = 26
N_WEEKS_POST  = 26
N_WEEKS_TOTAL = N_WEEKS_PRE + N_WEEKS_POST

def six_month_to_weekly(p):
    """Convert a 6-month cumulative churn rate to a weekly hazard."""
    return 1 - (1 - p) ** (1 / N_WEEKS_PRE)

baseline_weekly = six_month_to_weekly(TRUE_BASELINE)

# Each customer's post-renewal weekly hazard is set by their cohort.
A = customers['promo_expired'].values
B = customers['initiative_complete'].values
six_month_churn_post = (
    TRUE_BASELINE
    + TRUE_PROMO       * A
    + TRUE_INITIATIVE  * B
    + TRUE_INTERACTION * A * B
)
post_weekly_hazard = six_month_to_weekly(six_month_churn_post)

# Build (N x weeks) hazard matrix: pre-renewal = baseline, post = cohort-specific.
hazards = np.full((N, N_WEEKS_TOTAL), baseline_weekly)
hazards[:, N_WEEKS_PRE:] = post_weekly_hazard[:, None]

# Draw weekly churn events. Customers churn at most once (absorbing state).
draws           = RNG.uniform(size=(N, N_WEEKS_TOTAL))
would_churn     = draws < hazards
has_churn       = would_churn.any(axis=1)
first_churn_idx = np.where(has_churn, would_churn.argmax(axis=1), N_WEEKS_TOTAL)

# Customer-level 6-month outcome (used for Methods 2 and 3)
post_churn_idx = np.where(
    (first_churn_idx >= N_WEEKS_PRE) & (first_churn_idx < N_WEEKS_TOTAL),
    first_churn_idx, N_WEEKS_TOTAL
)
customers['churned'] = (post_churn_idx < N_WEEKS_TOTAL).astype(int)

print('Realized 6-month churn rates by cohort:')
print(customers.groupby(['promo_expired', 'initiative_complete'])['churned'].mean())

def build_cohort_week_panel(customers_df, first_churn_idx, has_churn):
    """
    Aggregate customer-week observations into (cohort, week) cells.
    A cohort is the (promo_expired, initiative_complete) combination,
    so there are 4 cohorts in total.
    """
    weeks = np.arange(-N_WEEKS_PRE, N_WEEKS_POST)
    A_arr = customers_df['promo_expired'].values
    B_arr = customers_df['initiative_complete'].values
    cohort_id = A_arr * 2 + B_arr   # encode 4 cohorts as 0, 1, 2, 3

    # at_risk[i, w] = 1 if customer i is still at risk in week w (no prior churn)
    week_idx    = np.arange(N_WEEKS_TOTAL)
    at_risk_mat = week_idx[None, :] <= first_churn_idx[:, None]
    churned_mat = (week_idx[None, :] == first_churn_idx[:, None]) & has_churn[:, None]

    rows = []
    for c in range(4):
        mask = cohort_id == c
        if mask.sum() == 0:
            continue
        a_val, b_val = (c >> 1) & 1, c & 1
        ar_by_week = at_risk_mat[mask].sum(axis=0)
        ch_by_week = churned_mat[mask].sum(axis=0)
        for w_idx, week in enumerate(weeks):
            if ar_by_week[w_idx] > 0:
                rows.append({
                    'week':       int(week),
                    'A':          int(a_val),     # promo_expired
                    'B':          int(b_val),     # initiative_complete
                    'at_risk':    int(ar_by_week[w_idx]),
                    'churns':     int(ch_by_week[w_idx]),
                    'churn_rate': ch_by_week[w_idx] / ar_by_week[w_idx],
                    'post':       int(week >= 0),
                })
    return pd.DataFrame(rows)

panel = build_cohort_week_panel(customers, first_churn_idx, has_churn)
print(f"Panel: {len(panel)} rows ({panel['week'].nunique()} weeks x "
      f"{panel.groupby(['A','B']).ngroups} cohorts)")
panel.head()

churn_by_cohort = customers.groupby(['promo_expired', 'initiative_complete'])['churned'].mean() * 100
labels = ['Neither\n(baseline)', 'Promo\nexpiry only',
          'Initiative\ncomplete only', 'Both forces\n(observed)']
values = [
    churn_by_cohort.loc[(0, 0)],
    churn_by_cohort.loc[(1, 0)],
    churn_by_cohort.loc[(0, 1)],
    churn_by_cohort.loc[(1, 1)],
]
expected = values[1] + values[2] - values[0]   # additive expectation
surplus  = values[3] - expected                # interaction surplus

fig, ax = plt.subplots(figsize=(7.5, 4.0))
bars = ax.bar(labels, values, color=[GRAY, BLUE, ORANGE, RED],
              width=0.52, edgecolor='white', linewidth=1.4, alpha=0.88)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
            f'{val:.1f}%', ha='center', fontweight='bold', color=DARK)

ax.annotate('', xy=(3, values[3]), xytext=(3, expected),
            arrowprops=dict(arrowstyle='<->', color=RED, lw=1.8))
ax.text(3.28, (values[3] + expected) / 2,
        f'+{surplus:.1f} pp\ninteraction\nsurplus',
        color=RED, fontsize=8.5, va='center', fontweight='bold')
ax.axhline(expected, color=GRAY, linestyle='--', linewidth=1, alpha=0.7)
ax.text(0.01, expected + 0.3, f'Additive expectation ({expected:.1f}%)',
        fontsize=8, color=GRAY)

ax.set_ylabel('6-month churn rate (%)')
ax.set_ylim(0, max(values) * 1.25)
ax.set_title('Churn rate by condition at renewal',
             fontsize=11, fontweight='bold', color=DARK, pad=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('figures/fig1_churn_by_condition.png', dpi=160, bbox_inches='tight')
plt.show()

# 'post * A * B' expands to: post, A, B, post:A, post:B, A:B, post:A:B.
# Weighting by at_risk gives bigger cohort-weeks more influence.
did_model = smf.wls(
    'churn_rate ~ post * A * B',
    data    = panel,
    weights = panel['at_risk'],
).fit(cov_type='HC3')   # heteroskedasticity-robust standard errors

print(did_model.summary().tables[1])

# Coefficients to read:
#   post:A   = promo shock when the initiative is still ongoing
#   post:B   = initiative shock when the promo has not expired
#   post:A:B = additional churn when both forces hit in the same week
print('\nKey DiD coefficients:')
for term in ['post:A', 'post:B', 'post:A:B']:
    coef, se = did_model.params[term], did_model.bse[term]
    print(f'  {term:12s}: {coef:+.4f}  (SE {se:.4f})')

# Restrict to initiative_complete == 0 to isolate the promo effect.
event_panel = panel[panel['B'] == 0].copy()

# Pivot so each column is one cohort (A=0 vs A=1), each row is one week.
pivot = event_panel.pivot_table(index='week', columns='A', values='churn_rate')

# Center each cohort's hazard at its own pre-renewal mean.
# This makes the chart show "relative" hazard, so both cohorts start near zero.
pre_period = pivot[pivot.index < 0]
pivot_centered = pivot - pre_period.mean()

# Plot the two lines: promo-expired cohort vs comparison cohort.
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Renewal date')
ax.axhline(0, color=GRAY, linewidth=0.8, alpha=0.5)
ax.plot(pivot_centered.index, pivot_centered[1],
        color=BLUE, linewidth=1.8, label='Promo-expired cohort')
ax.plot(pivot_centered.index, pivot_centered[0],
        color=ORANGE, linewidth=1.8, linestyle='--', label='Comparison cohort')

ax.set_xlabel('Weeks relative to renewal')
ax.set_ylabel('Weekly churn hazard (relative)')
ax.set_title('Event study: parallel trends diagnostic',
             fontsize=11, fontweight='bold', color=DARK, pad=10)
ax.legend(fontsize=9, framealpha=0, loc='upper left')
ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('figures/fig2_event_study.png', dpi=160, bbox_inches='tight')
plt.show()

# Customer-level regression. Outcome: 1 if customer churned within 6 months.
# np.log1p(x) = log(1 + x); used to control for skewed dollar/count covariates
# (annual revenue, seat counts) so a few large customers do not dominate.
# The * operator below expands to: main effects of A and B AND their interaction.
interaction_model = smf.ols(
    'churned ~ promo_expired * initiative_complete'
    '       + np.log1p(arr_usd) + np.log1p(n_seats)',
    data=customers,
).fit(cov_type='HC3')   # HC3 = heteroskedasticity-robust standard errors

print(interaction_model.summary().tables[1])

b1 = interaction_model.params['promo_expired']
b2 = interaction_model.params['initiative_complete']
b3 = interaction_model.params['promo_expired:initiative_complete']

print('\nConditional effects:')
print(f'  Promo expiry, initiative ongoing:     b1       = {b1:+.4f}')
print(f'  Promo expiry, initiative complete:    b1 + b3  = {b1 + b3:+.4f}')
print(f'  Initiative complete, promo ongoing:   b2       = {b2:+.4f}')
print(f'  Initiative complete, promo expired:   b2 + b3  = {b2 + b3:+.4f}')

incremental = [
    values[1] - values[0],                  # promo only
    values[2] - values[0],                  # initiative only
    values[1] + values[2] - 2 * values[0],  # additive expectation
    values[3] - values[0],                  # observed both
]
labels3 = ['Promo expiry\nonly (A)', 'Initiative\ncomplete only (B)',
           'Expected\n(additive)', 'Observed\n(A + B)']

fig, ax = plt.subplots(figsize=(7.5, 4.0))
bars = ax.bar(labels3, incremental, color=[BLUE, ORANGE, GRAY, RED],
              width=0.52, edgecolor='white', linewidth=1.4, alpha=0.88)
for bar, val in zip(bars, incremental):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
            f'+{val:.1f} pp', ha='center', fontweight='bold', color=DARK)
ax.annotate('', xy=(3, incremental[3]), xytext=(3, incremental[2]),
            arrowprops=dict(arrowstyle='<->', color=RED, lw=1.8))
ax.text(3.28, (incremental[3] + incremental[2]) / 2,
        f'+{incremental[3] - incremental[2]:.1f} pp\nsurplus',
        color=RED, fontsize=8.5, va='center', fontweight='bold')
ax.axhline(incremental[2], color=GRAY, linestyle='--', linewidth=1, alpha=0.7)
ax.set_ylabel('Incremental churn above baseline (pp)')
ax.set_ylim(0, max(incremental) * 1.25)
ax.set_title('Interaction effect: joint impact exceeds sum of parts',
             fontsize=11, fontweight='bold', color=DARK, pad=10)
ax.yaxis.grid(True, linestyle='--', alpha=0.4); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('figures/fig3_interaction.png', dpi=160, bbox_inches='tight')
plt.show()

# Coalition values: incremental churn (pp) caused by each subset of drivers.
v = {
    frozenset():                    0,                       # neither
    frozenset(['promo']):           round(incremental[0], 1),
    frozenset(['init']):            round(incremental[1], 1),
    frozenset(['promo', 'init']):   round(incremental[3], 1),
}
print(f'Coalition values: {dict(v)}')

# 'players' = the drivers we are allocating credit across.
# For each ordering of players, each player's 'marginal contribution' is
# how much the coalition value grows when that player joins.
# Shapley value = average marginal contribution across all orderings.
def shapley_values(v, players):
    n   = len(players)
    phi = {p: 0.0 for p in players}      # accumulator for each player
    for perm in permutations(players):    # try every ordering
        coalition = frozenset()           # start with no drivers active
        for player in perm:
            # how much does the coalition value grow when this player joins?
            marginal     = v[coalition | {player}] - v[coalition]
            phi[player] += marginal
            coalition    = coalition | {player}
    # average across all n! orderings
    return {p: round(phi[p] / math.factorial(n), 2) for p in players}

phi = shapley_values(v, ['promo', 'init'])
print(f'Shapley allocation: {phi}')
print(f'Sum: {sum(phi.values()):.2f} pp  (should equal v(promo, init))')

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6))

ax1 = axes[0]
cvals = [v[frozenset(['promo'])], v[frozenset(['init'])], v[frozenset(['promo','init'])]]
b3_bars = ax1.bar(['v(A)', 'v(B)', 'v(A, B)'], cvals,
                  color=[BLUE, ORANGE, RED], width=0.45,
                  edgecolor='white', linewidth=1.4, alpha=0.88)
for bar, val in zip(b3_bars, cvals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.2,
             f'{val:.1f} pp', ha='center', fontweight='bold', color=DARK)
ax1.set_ylabel('Incremental churn (pp)')
ax1.set_title('Coalition values', fontsize=10.5, fontweight='bold', color=DARK)
ax1.set_ylim(0, max(cvals) * 1.2)
ax1.yaxis.grid(True, linestyle='--', alpha=0.4); ax1.set_axisbelow(True)

ax2 = axes[1]
ax2.barh(0, phi['promo'], color=BLUE, height=0.4, alpha=0.88,
         edgecolor='white', linewidth=1.5, label='Promo expiry (A)')
ax2.barh(0, phi['init'], left=phi['promo'], color=ORANGE, height=0.4, alpha=0.88,
         edgecolor='white', linewidth=1.5, label='Initiative complete (B)')
ax2.text(phi['promo'] / 2, 0, f"{phi['promo']:.1f} pp", ha='center', va='center',
         color='white', fontweight='bold')
ax2.text(phi['promo'] + phi['init'] / 2, 0, f"{phi['init']:.1f} pp",
         ha='center', va='center', color='white', fontweight='bold')
ax2.set_xlim(0, (phi['promo'] + phi['init']) * 1.15)
ax2.set_ylim(-0.8, 0.8)
ax2.set_title(f"Shapley allocation\n(sums to {sum(phi.values()):.1f} pp total)",
              fontsize=10.5, fontweight='bold', color=DARK)
ax2.legend(fontsize=8.5, framealpha=0, loc='lower center',
           bbox_to_anchor=(0.5, -0.55))
ax2.axis('off')

plt.tight_layout(pad=1.5)
plt.savefig('figures/fig4_shapley.png', dpi=160, bbox_inches='tight')
plt.show()

def ltv(monthly_churn, monthly_mrr, horizon=24):
    """
    LTV = expected revenue per customer over a fixed horizon (months).
    survival[m] = probability customer is still subscribed in month m.
    Returns undiscounted LTV. For discounted LTV, multiply by (1 + r) ** -m.
    """
    months   = np.arange(horizon)
    survival = (1 - monthly_churn) ** months
    return (survival * monthly_mrr).sum()

# Convert 6-month churn rates into monthly churn rates.
# (1 - p)^(1/6) is the monthly survival rate that compounds to (1 - p) in 6 months.
baseline_monthly = 1 - (1 - 0.08) ** (1/6)   # 0.0138 monthly churn
treated_monthly  = 1 - (1 - 0.22) ** (1/6)   # 0.0406 monthly churn

old_mrr, new_mrr = 1_000, 1_130   # 13% price increase

baseline_ltv = ltv(baseline_monthly, old_mrr)
treated_ltv  = ltv(treated_monthly,  new_mrr)

print(f'Baseline 2-year LTV: ${baseline_ltv:>10,.0f}')
print(f'Treated  2-year LTV: ${treated_ltv:>10,.0f}')
print(f'Net change:          ${treated_ltv - baseline_ltv:>10,.0f}')

# Breakeven: what new MRR would restore the baseline 2-year LTV?
price_grid = np.linspace(1_000, 1_600, 1_000)
ltv_grid   = np.array([ltv(treated_monthly, p) for p in price_grid])
breakeven  = price_grid[np.searchsorted(ltv_grid, baseline_ltv)]

print(f'\nBreakeven MRR: ${breakeven:,.0f}  '
      f'({(breakeven / old_mrr - 1) * 100:.0f}% increase vs the 13% that shipped)')
