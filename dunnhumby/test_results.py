"""Test results tracking (following sim/ pattern).

This module provides a simple test tracking class for reporting
test outcomes in the Dunnhumby integration test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SimulationResults:
    """
    Track simulation/test results.

    This class follows the pattern from sim/test_garp_consistency.py
    for consistent test reporting.

    Attributes:
        name: Name of the test suite/category
        tests_run: Total number of tests executed
        tests_passed: Number of tests that passed
        failures: List of failure descriptions
    """

    name: str
    tests_run: int = 0
    tests_passed: int = 0
    failures: List[str] = field(default_factory=list)

    def record(self, test_name: str, passed: bool, message: str = "") -> None:
        """
        Record a test result.

        Args:
            test_name: Name/description of the test
            passed: Whether the test passed
            message: Additional message (typically for failures)
        """
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
            print(f"  [PASS] {test_name}")
        else:
            failure_msg = f"{test_name}: {message}" if message else test_name
            self.failures.append(failure_msg)
            print(f"  [FAIL] {test_name}: {message}")

    def summary(self) -> bool:
        """
        Print summary and return overall success.

        Returns:
            True if all tests passed, False otherwise
        """
        print(f"\n{self.name}: {self.tests_passed}/{self.tests_run} tests passed")
        if self.failures:
            print("Failures:")
            for f in self.failures:
                print(f"  - {f}")
        return len(self.failures) == 0

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed."""
        return len(self.failures) == 0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as percentage."""
        if self.tests_run == 0:
            return 0.0
        return 100.0 * self.tests_passed / self.tests_run
