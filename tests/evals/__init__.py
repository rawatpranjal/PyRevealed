"""
EVALs Suite for PyRevealed - Brutal Tests Designed to Break the Library

Philosophy:
    These tests are MEANT TO FAIL. The goal is to expose weaknesses, not prove correctness.
    - A failing eval = discovered vulnerability = good
    - Tests document known limitations and edge cases
    - Use pytest.mark.xfail for known issues (expected failures)
    - Track which evals start passing after fixes

Run tests:
    pytest tests/evals/ -v                    # Run all evals
    pytest tests/evals/ --tb=no -q           # Quick summary of failures
    pytest tests/ --ignore=tests/evals/      # Run regular tests only
"""
