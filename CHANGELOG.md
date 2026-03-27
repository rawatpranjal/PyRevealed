# Changelog

## [0.5.3] - 2026-03-27

### Added
- `rp.analyze()` one-liner API — auto-detects long, wide, and menu formats from a DataFrame
- `nan_policy` parameter ("raise"/"warn"/"drop") for `analyze()` and `BehaviorLog`
- `load_pakistan()`, `load_favorita()`, `load_taobao()` dataset loaders
- 6-dataset e-commerce benchmark: 162K users, 14 tasks, RP features add 0–0.7% AUC
- LLM benchmark v2 Stage 1: 3,750 decisions, per-vignette SARP pass rates 60–100%
- Smart error messages with fuzzy column-name suggestions and type-conversion hints
- Python 3.13 wheels for all platforms
- Rust build now optional for source installs — pure-Python fallback via setuptools-rust

### Fixed
- Linux/Windows install broken — v0.5.2 only shipped macOS wheel

## [0.5.2] - 2026-03-26

### Changed
- Front page hero GIFs redesigned as single-panel narrative animations with breathing gaps
- Front page text rewritten in plain English with GIF captions and citations

## [0.5.1] - 2026-03-26

### Fixed
- ReadTheDocs builds broken since maturin migration — mock Rust extension in autodoc
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
- `load_demo()` synthetic dataset — 100 consumers, zero setup, deterministic
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
- Rust engine (`rpt-core`) with Rayon parallelism — 18-100x faster than Python
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
