# Changelog

## [Unreleased]

### Changed
- CI: add GitHub Actions workflow `.github/workflows/docs.yml` to always build Sphinx HTML on every push and PR; uploads HTML as an artifact. Keeps RTD as the publisher while catching build issues early.
- Front page LLM summary table: clarified column names (no abbreviations), split deterministic vs stochastic SARP into separate columns, expanded IIA into separate deterministic/stochastic columns, and added RUM pass rate (%). Also clarified footnote definitions for all metrics. Source numbers pulled from docs page `docs/budget/app_llm_benchmark.rst` to ensure consistency.
- Removed stock illustrations from LLM consistency docs: dropped robot graphic from `budget/app_llm_benchmark` and `benchmarks` cards; removed unused `_static/app_llm_stock.svg`. Also removed decorative hero image from `budget/app_llm_alignment` to keep the page focused on real outputs.
- Replaced Benchmarks hub card image with real results thumbnail (`_static/app_llm_benchmark_summary.png`).
 - Menus page GIFs: replaced raw HTML <img> with a robust list-table of images to ensure they render reliably on RTD.
 - Slightly sped up all docs GIF animations (~1.25× faster) to feel snappier. No changes to CSS/JS text transitions; only GIF frame delays were reduced.
 - Further increased GIF speed for a total ~1.5× faster vs original (applied an additional 1.2× pass to existing GIF timings). Text transitions remain unchanged.

### Added
- Read the Docs config `.readthedocs.yaml` to ensure RTD builds consistently with pinned dependencies and without compiling the Rust extension.
- GitHub Actions workflow `docs.yml` to build Sphinx docs on every push/PR to `main` (catches doc issues before RTD).

## [0.5.13] - 2026-03-28

### Added
- E-commerce benchmarks: consolidate Taobao (Buy-Anchored) results into main index and benchmarks.

## [0.5.13] - 2026-03-28

### Added
- E-commerce benchmarks: consolidate Taobao (Buy-Anchored) results into main index and benchmarks.

## [0.5.9] - 2026-03-28

### Changed
- Performance page rewritten into concise paragraphs; consolidated RTD plot generation under `docs/perf/` with a CLI (`python -m docs.perf.cli`) and migrated legacy script to delegate. Figures write to `docs/_static/perf_*.png`.

### Fixed
- **Base install safety** - `import prefgraph` no longer crashes without `pandas`. Four dataset loaders (_retailrocket, _rees46, _taobao, _tenrec) had bare `import pandas as pd` at module level; now lazily imported via wrapper functions in `datasets/__init__.py`.
- **compute_mpi() function API** - was silently returning 0.0 on GARP-violating data. Root cause: Rust call had wrong argument count (missing `network=False`), causing TypeError caught by bare `except`, falling to Python fallback which had a separate `is_garp` gate bug. Fixed arg count per engine.py call signature. Ref: Echenique, Lee & Shum (2011) JPE 119(6), Eq. (2).
- **Engine VEI** - `Engine(metrics=['vei'])` returned `vei_mean=1.0` on inconsistent data. Neither the Rust backend nor the Python fallback called `compute_vei()`. Added Python VEI computation to both paths. Ref: Varian (1990) J. Econometrics; Mononen (2023) "Computing Measures of Rationality".
- **BehavioralAuditor HM** - `summary().houtman_maks_result` was `None` on consistent logs. Now always computed (fast-exits trivially with fraction=0.0). Ref: Heufer & Hjertstrand (2015) "Consistent Subsets".

## [0.5.8] - 2026-03-28

### Fixed
- **VEI objective sign** - `compute_vei()` LP was maximizing sum(e_i) instead of minimizing, trivially returning e_i=1.0 for all observations regardless of GARP violations. Now uses direct R constraints and minimize direction, producing meaningful per-observation efficiency scores. Ref: Varian (1990) J. Econometrics; Smeulders et al. (2014) ACM Trans. Econ. Comp.
- **HM greedy SCC** - `_houtman_maks_greedy()` used `find_sccs(R)` (direct relation) instead of `find_sccs(R_star)` (transitive closure). Purely transitive GARP violations produced all-singleton SCCs, causing the greedy FVS to return 0 removals even when violations existed. Ref: Houtman & Maks (1985); Smeulders et al. (2014) Theorem 5.1.
- **Production GARP** - Python `test_profit_maximization()` checked `R_star[i,j] AND R_star[j,i]` (cycle only) instead of `R_star[i,j] AND P[j,i]` (proper GARP condition). Now matches the Rust implementation. Ref: Varian (1984) Econometrica; Chambers & Echenique (2016) Ch 15.
- **EngineResult defaults** - `is_harp`, `utility_success`, `hm_consistent`, `hm_total` now default to `None` (not `False`/`0`) when not computed. DataFrames show NaN instead of misleading failure values.

## [0.5.7] - 2026-03-28

### Fixed
- Python fallback engine now computes HARP and utility_success (were always False without Rust)

## [0.5.6] - 2026-03-28

Re-release of 0.5.5 with multi-platform wheels (Linux manylinux2_28, macOS x86+arm, Windows, Python 3.10–3.13). v0.5.5 only shipped a macOS cp311 wheel.

## [0.5.5] - 2026-03-28

### Fixed
- **Houtman-Maks ILP rewritten** - replaced Afriat Big-M formulation (broken: M too small, λ bounds infeasible) with Demuynck & Rehbeck (2023) Corollary 2. Uses fixed parameters α, δ, ε with clean data-derived bounds. Exact on all tested counterexamples.
- **Engine now uses exact ILP** for T ≤ 200 (`houtman_maks_exact`), greedy FVS for T > 200. Previously always used greedy, which over-removed observations.
- **HARP severity dropped** - `max_cycle_product` always returns 1.0. HARP is a binary test per Varian (1983) and Chambers & Echenique (2016, Thm 4.2); no severity metric is defined in the literature. Previous implementations (FW diagonal, 2-cycle patch) were approximations of a non-existent quantity.
- **Python HM ILP** - same Big-M fix as Rust (lambda bounds, M computation)

## [0.5.4] - 2026-03-27

### Fixed
- `validate_consistency` docstring referenced non-existent `is_valid` and `inconsistencies` fields - now correctly documents `is_consistent` and `violations`
- Install command in quickstart and README changed from `pip install prefgraph` to `pip install "prefgraph[datasets]"` - beginners no longer hit missing pandas on first run
- HM (Houtman-Maks) in Python fallback returned `hm_consistent=0, hm_total=0` - now computes real values
- `EngineResult` docstring described `hm_consistent / hm_total` as "noise fraction" - corrected to "rationalizable fraction"

### Added
- `HoutmanMaksResult.efficiency` property - returns fraction of rationalizable observations (`1 - fraction`), matching the polarity of `compute_menu_efficiency().efficiency_index`
- User_id note in quickstart - clarifies that `analyze()` output has `user_id` as DataFrame index

## [0.5.3] - 2026-03-27

### Added
- `rp.analyze()` one-liner API - auto-detects long, wide, and menu formats from a DataFrame
- **Parquet streaming**: `rp.analyze("data.parquet", ...)` and `Engine.analyze_parquet()` - stream datasets larger than RAM with bounded O(chunk_size) memory
- **Rust-native Parquet pipeline**: `analyze_parquet_file()` reads Parquet, groups by user, and feeds directly to Rayon - eliminates Python from the hot path (requires `--features parquet`)
- `prefgraph.io.ParquetUserIterator` - streaming row-group reader for wide and long formats
- `prefgraph.io.prepare_parquet()` - sort and rewrite datasets for optimal streaming
- `nan_policy` parameter ("raise"/"warn"/"drop") for `analyze()` and `BehaviorLog`
- `load_pakistan()`, `load_favorita()`, `load_taobao()` dataset loaders
- 6-dataset e-commerce benchmark: 162K users, 14 tasks, RP features add 0–0.7% AUC
- LLM benchmark v2 Stage 1: 3,750 decisions, per-vignette SARP pass rates 60–100%
- Smart error messages with fuzzy column-name suggestions and type-conversion hints
- Python 3.13 wheels for all platforms
- Rust build now optional for source installs - pure-Python fallback via setuptools-rust

### Fixed
- Linux/Windows install broken - v0.5.2 only shipped macOS wheel

## [0.5.2] - 2026-03-26

### Changed
- Front page hero GIFs redesigned as single-panel narrative animations with breathing gaps
- Front page text rewritten in plain English with GIF captions and citations

## [0.5.1] - 2026-03-26

### Fixed
- ReadTheDocs builds broken since maturin migration - mock Rust extension in autodoc
- PyPI homepage URL now points to ReadTheDocs instead of GitHub
- Linux CI: pin to Python 3.10-3.12 (PyO3 0.22 max), install libclang for HiGHS bindgen
- Escaped pipe characters in RST list-tables causing Sphinx build errors

### Changed
- Front page hero GIFs redesigned: single-panel narrative animations (14s each) replacing 3-panel 34s versions
- Budget GIF walks through violation detection → CCEI measurement with concrete axis labels (Apples/Oranges)
- Menu GIF uses concrete items (Laptop/Tablet/Phone) instead of abstract A/B/C letters
- Front page text rewritten in plain English; removed jargon from intro and "Why RP?" sections
- Added GIF captions explaining what each animation shows, with citations
- Simplified pipeline diagram to plain English with function names

### Added
- `load_demo()` synthetic dataset - 100 consumers, zero setup, deterministic
- `EngineResult.summary()` and `MenuResult.summary()` methods
- Field-level documentation for `EngineResult` and `MenuResult` dataclasses
- Quickstart page in documentation
- "Which API?" guidance in README
- Application pages: grocery rationality, LLM alignment, recommender systems
- Animated algorithm visualizations (Floyd-Warshall, SCC Tarjan, engine throughput)
- Hero images for budget and menu analysis pages
- Stock images for application pages
- `CHANGELOG.md`

### Changed
- README reordered: quick example first, architecture details moved to docs
- Documentation navigation restructured into data-type subdirectories
- Single-word nav tab names (Budgets, Menus, Algorithms, Performance, etc.)

## [0.5.0] - 2026-03-25

### Added
- Rust engine (`rpt-core`) with Rayon parallelism - 18-100x faster than Python
- `Engine` class for batch analysis of millions of users
- `Engine.analyze_menus()` for discrete/menu choice data
- `Engine.build_graph()` for deep per-user preference graph construction
- VEI (Varian Efficiency Index) metric
- Houtman-Maks consistency fraction via Engine
- HARP (homothetic axiom) via Engine
- Utility recovery via Afriat LP in Engine
- Cross-platform CI release workflow (macOS, Linux, Windows)
- Result dataclasses: `EngineResult`, `MenuResult`
- Backend parity tests (`tests/test_backend_parity.py`)

### Changed
- Build backend switched from setuptools to maturin
- `result.py` modularized into `core/results/` subpackage
