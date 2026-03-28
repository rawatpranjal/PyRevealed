# Open E‑Commerce (Amazon) — EDA: Budget vs Menu Decision

Date: 2026-03-28

Summary: The Open E‑Commerce Amazon dataset in this repo is purchase‑history data aggregated to monthly category spend per user. It does not contain exposures, impressions, search queries, or per‑session menus. Therefore, a budget‑choice (revealed preference) analysis is possible and appropriate; a menu‑choice analysis is not supported by the available signals.

## Signals Detected (from loader)

- Observed columns: `Order Date`, `Category` (mapped to groups), `Quantity`, `Purchase Price Per Unit`, `Survey ResponseID`.
- Derived by loader:
  - Month `period` from `Order Date` (monthly choice occasions).
  - Category mapping to ~50 grouped categories.
  - Prices per user: realized prices used where the user purchased a category in a month; otherwise filled by market median (period×category). This mirrors the H&M loader pattern to preserve individual price variation.
- Absent: exposures/impressions, views, search queries, ranks, or any explicit menus.

File references:
- CSV read: `src/prefgraph/datasets/_open_ecommerce.py:106`
- Date parsing/periods: `src/prefgraph/datasets/_open_ecommerce.py:109`, `src/prefgraph/datasets/_open_ecommerce.py:111`
- Category mapping: `src/prefgraph/datasets/_open_ecommerce.py:114`
- Filters (valid prices/quantities): `src/prefgraph/datasets/_open_ecommerce.py:118`–`src/prefgraph/datasets/_open_ecommerce.py:119`
- Price oracle (median per period×category): `src/prefgraph/datasets/_open_ecommerce.py:130`
- User-level realized prices with oracle fallback: `src/prefgraph/datasets/_open_ecommerce.py:150`
- User identifier: `src/prefgraph/datasets/_open_ecommerce.py:139`

## Feasibility

- Budget‑choice: YES — For each user and month t, the loader constructs a price vector p_t (categories) and a quantity vector x_t (purchases). Standard RP/GARP tools apply to the sequence {(p_t, x_t)}.
- Menu‑choice: NO — Menus of alternatives are not observed (no impressions/views). There is no defensible way to reconstruct simultaneous choice sets from this dataset alone.

## Plausibility Considerations (Budget)

- Partial budget: Amazon is a subset of total consumption. Budgets m_t are implicit (m_t = p_t·x_t) and represent platform‑specific spend, which is sufficient for RP tests but not “total income”.
- Prices: For unpurchased categories in a month, prices are imputed from market medians (price oracle). This is standard for panel RP but limits individual‑level price variation.
- Goods definition: Goods are categories (≈50). Within‑category substitution across SKUs is not observed; analysis is at category level.

## Recommended Path (Budget‑Choice)

1) Place `amazon-purchases.csv` in one of:
   - `~/.prefgraph/data/open_ecommerce/`
   - `src/../datasets/open_ecommerce/data/`
   - Or pass a custom `data_dir` to the loader.

2) Load and analyze:

```python
import prefgraph as rp
from prefgraph.datasets import load_open_ecommerce

panel = load_open_ecommerce(min_observations=5, top_n_categories=50)
results = rp.Engine(metrics=["garp", "ccei", "mpi"]).analyze_panel(panel)
print(results.head())
```

3) Interpretable summaries (per user):
- GARP pass rate, Afriat Efficiency Index (CCEI), Money Pump Index (MPI)
- Price/quantity summary, budget waste, welfare bounds

## If You Need Menu‑Choice

- Not supported by this dataset. You would need exposure/search logs (impressions with positions, timestamps, and the chosen item) to form credible menus. Without such logs, any synthetic menu would be speculative and likely invalid for menu‑choice inference.

## Price Variation + RP Diagnostics

Use `tools/budget_variation.py` to quantify price variation and RP test power:

```
python tools/budget_variation.py --dataset open_ecommerce \
  --data-dir /path/to/open_ecommerce --n-users 1000
```

Outputs:
- Share of users with multiple distinct price vectors over time
- Per-user mean price CV across categories (percentiles)
- Category-level unique monthly price counts
- GARP pass rate and CCEI/MPI percentiles
- Correlation between price variation and CCEI

