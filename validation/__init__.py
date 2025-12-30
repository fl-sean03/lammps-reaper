"""Validation subpackage for lammps_reaper.

This package contains multi-level validation for LAMMPS input decks:
- L0: Placeholder validation (unresolved template variables)
- L1: Syntax + Physics validation (LAMMPS command syntax + physics parameters)
- L2: Engine validation (LAMMPS execution check with 0 steps)
- L3: Minimal execution validation (LAMMPS run with ~20 steps to catch explosions)
"""

from pathlib import Path
from typing import List, Optional

from ..schemas import L0Result, L1Result, L2Result, L3Result, ValidationResult
from .l0_placeholders import validate_l0
from .l1_syntax import validate_l1
from .l2_engine import find_lammps_binary, validate_l2
from .l3_physics import validate_l3
from .file_utils import (
    parse_file_references,
    find_file_in_context,
    setup_working_directory,
    cleanup_working_directory,
)

__all__ = [
    "validate_l0",
    "validate_l1",
    "validate_l2",
    "validate_l3",
    "validate_deck",
    "find_lammps_binary",
    # File utilities
    "parse_file_references",
    "find_file_in_context",
    "setup_working_directory",
    "cleanup_working_directory",
]


def validate_deck(
    content: str,
    lammps_binary: Optional[Path] = None,
    context_files: Optional[List[Path]] = None,
) -> ValidationResult:
    """Run all validation levels on a LAMMPS deck.

    Performs multi-level validation:
    - L0: Placeholder detection ({{VAR}}, <VAR>, TODO:, FIXME:)
    - L1: LAMMPS syntax + physics parameter validation
    - L2: Engine acceptance check (run 0 steps)
    - L3: Minimal execution check (run ~20 steps to catch explosions)

    Args:
        content: The LAMMPS deck content to validate.
        lammps_binary: Optional path to LAMMPS binary for L2/L3 validation.
        context_files: Optional list of files to copy to working directory
                       for L2/L3 validation (e.g., data files, potentials).

    Returns:
        ValidationResult containing results from all validation levels.
    """
    issues: List[str] = []

    # L0: Placeholder validation
    l0_result = validate_l0(content)
    if not l0_result.passed:
        issues.append(f"L0: {l0_result.unresolved_count} unresolved placeholder(s) found")

    # L1: Syntax + Physics validation
    l1_result = validate_l1(content)
    if not l1_result.passed:
        error_count = len(l1_result.syntax_errors)
        physics_count = len([w for w in l1_result.physics_warnings if "Negative" in w])
        issues.append(f"L1: {error_count} syntax error(s), {physics_count} critical physics error(s)")

    # L2: Engine acceptance (0 steps)
    l2_result = validate_l2(content, lammps_binary, context_files=context_files)
    if not l2_result.passed:
        issues.append("L2: LAMMPS engine acceptance failed")

    # L3: Minimal execution (~20 steps)
    l3_result = validate_l3(content, lammps_binary, context_files=context_files)
    if not l3_result.passed:
        issues.append("L3: LAMMPS minimal execution failed")

    # Overall pass requires L0, L1, L2, and L3 to pass
    overall_passed = (
        l0_result.passed
        and l1_result.passed
        and l2_result.passed
        and l3_result.passed
    )

    return ValidationResult(
        overall_passed=overall_passed,
        l0=l0_result,
        l1=l1_result,
        l2=l2_result,
        l3=l3_result,
        issues=issues,
    )
