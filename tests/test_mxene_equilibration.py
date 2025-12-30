"""End-to-end tests for MXene equilibration deck generation.

This module tests the LAMMPS Reaper's ability to generate equilibration
workflows for MXene 2D materials with and without context files.

Test 1: Data file only - LLM must infer force field from data file
Test 2: Data file + context files - LLM has full workflow examples
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from lammps_reaper import generate_deck, validate_deck, ReaperInput

# Asset paths
ASSETS_DIR = Path(__file__).parent.parent.parent / "lammps_reaper" / "assets"
DATA_FILE = ASSETS_DIR / "equil_nvt_dry.data"
MINIMIZE_INP = ASSETS_DIR / "minimize_variable.inp"
NPT_INP = ASSETS_DIR / "equilibration_npt_variable.inp"
NVT_INP = ASSETS_DIR / "equilibration_nvt_variable.inp"
WORKFLOW_SCRIPT = ASSETS_DIR / "run_equilibration_variable_data.sh"


@dataclass
class DeckAnalysis:
    """Analysis results for a generated LAMMPS deck."""

    # Basic structure
    has_units_real: bool = False
    has_atom_style_full: bool = False
    has_read_data: bool = False
    has_boundary_ppp: bool = False

    # Force field
    has_pair_style_lj_coul: bool = False
    has_bond_style_harmonic: bool = False
    has_angle_style_harmonic: bool = False
    has_dihedral_style: bool = False
    has_improper_style: bool = False
    has_kspace_pppm: bool = False

    # Simulation phases
    has_minimize: bool = False
    has_npt_fix: bool = False
    has_nvt_fix: bool = False

    # Settings
    has_timestep: bool = False
    timestep_value: Optional[float] = None
    has_thermo: bool = False
    has_run_command: bool = False
    total_run_steps: int = 0

    # Temperature/Pressure
    temperature: Optional[float] = None
    pressure: Optional[float] = None

    # Output
    has_dump: bool = False
    has_restart: bool = False
    has_write_data: bool = False

    # Quality metrics
    deck_length: int = 0
    comment_lines: int = 0

    def score(self) -> int:
        """Calculate quality score (0-100)."""
        points = 0

        # Essential (40 points)
        if self.has_units_real: points += 8
        if self.has_atom_style_full: points += 8
        if self.has_read_data: points += 8
        if self.has_pair_style_lj_coul: points += 8
        if self.has_kspace_pppm: points += 8

        # Force field completeness (20 points)
        if self.has_bond_style_harmonic: points += 4
        if self.has_angle_style_harmonic: points += 4
        if self.has_dihedral_style: points += 4
        if self.has_improper_style: points += 4
        if self.has_boundary_ppp: points += 4

        # Simulation phases (20 points)
        if self.has_minimize: points += 7
        if self.has_npt_fix: points += 7
        if self.has_nvt_fix: points += 6

        # Settings & output (20 points)
        if self.has_timestep: points += 4
        if self.has_thermo: points += 4
        if self.has_run_command: points += 4
        if self.has_dump or self.has_restart or self.has_write_data: points += 4
        if self.comment_lines >= 5: points += 4

        return min(points, 100)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Score: {self.score()}/100",
            "",
            "Essential:",
            f"  units real: {'✓' if self.has_units_real else '✗'}",
            f"  atom_style full: {'✓' if self.has_atom_style_full else '✗'}",
            f"  read_data: {'✓' if self.has_read_data else '✗'}",
            f"  pair_style lj/cut/coul/long: {'✓' if self.has_pair_style_lj_coul else '✗'}",
            f"  kspace pppm: {'✓' if self.has_kspace_pppm else '✗'}",
            "",
            "Force Field:",
            f"  bond_style harmonic: {'✓' if self.has_bond_style_harmonic else '✗'}",
            f"  angle_style harmonic: {'✓' if self.has_angle_style_harmonic else '✗'}",
            f"  dihedral_style: {'✓' if self.has_dihedral_style else '✗'}",
            f"  improper_style: {'✓' if self.has_improper_style else '✗'}",
            "",
            "Simulation Phases:",
            f"  minimize: {'✓' if self.has_minimize else '✗'}",
            f"  NPT fix: {'✓' if self.has_npt_fix else '✗'}",
            f"  NVT fix: {'✓' if self.has_nvt_fix else '✗'}",
            "",
            "Settings:",
            f"  timestep: {self.timestep_value if self.timestep_value else 'not set'}",
            f"  temperature: {self.temperature if self.temperature else 'not set'}",
            f"  total run steps: {self.total_run_steps}",
            "",
            f"Deck: {self.deck_length} chars, {self.comment_lines} comment lines",
        ]
        return "\n".join(lines)


def analyze_deck(content: str) -> DeckAnalysis:
    """Analyze a generated LAMMPS deck for quality metrics."""
    analysis = DeckAnalysis()
    analysis.deck_length = len(content)

    lines = content.split("\n")

    for line in lines:
        stripped = line.strip().lower()
        original = line.strip()

        # Count comments
        if stripped.startswith("#"):
            analysis.comment_lines += 1
            continue

        if not stripped:
            continue

        # Parse commands
        parts = stripped.split()
        if not parts:
            continue

        cmd = parts[0]

        # Units
        if cmd == "units" and len(parts) > 1:
            if parts[1] == "real":
                analysis.has_units_real = True

        # Atom style
        if cmd == "atom_style" and len(parts) > 1:
            if parts[1] == "full":
                analysis.has_atom_style_full = True

        # Boundary
        if cmd == "boundary" and len(parts) >= 4:
            if parts[1] == "p" and parts[2] == "p" and parts[3] == "p":
                analysis.has_boundary_ppp = True

        # Read data
        if cmd == "read_data":
            analysis.has_read_data = True

        # Pair style
        if cmd == "pair_style" and len(parts) > 1:
            rest = " ".join(parts[1:])
            if "lj/cut/coul" in rest or "lj/cut/coul/long" in rest:
                analysis.has_pair_style_lj_coul = True

        # Bond style
        if cmd == "bond_style" and len(parts) > 1:
            if "harmonic" in parts[1]:
                analysis.has_bond_style_harmonic = True

        # Angle style
        if cmd == "angle_style" and len(parts) > 1:
            if "harmonic" in parts[1]:
                analysis.has_angle_style_harmonic = True

        # Dihedral style
        if cmd == "dihedral_style":
            analysis.has_dihedral_style = True

        # Improper style
        if cmd == "improper_style":
            analysis.has_improper_style = True

        # Kspace
        if cmd == "kspace_style" and len(parts) > 1:
            if "pppm" in parts[1] or "ewald" in parts[1]:
                analysis.has_kspace_pppm = True

        # Minimize
        if cmd == "minimize" or cmd == "min_style":
            analysis.has_minimize = True

        # Fix NPT
        if cmd == "fix" and "npt" in stripped:
            analysis.has_npt_fix = True
            # Try to extract temperature
            temp_match = re.search(r"temp\s+([\d.]+)", stripped)
            if temp_match:
                analysis.temperature = float(temp_match.group(1))
            # Try to extract pressure
            press_match = re.search(r"iso\s+([\d.]+)", stripped)
            if press_match:
                analysis.pressure = float(press_match.group(1))

        # Fix NVT
        if cmd == "fix" and "nvt" in stripped and "npt" not in stripped:
            analysis.has_nvt_fix = True
            if not analysis.temperature:
                temp_match = re.search(r"temp\s+([\d.]+)", stripped)
                if temp_match:
                    analysis.temperature = float(temp_match.group(1))

        # Timestep
        if cmd == "timestep" and len(parts) > 1:
            analysis.has_timestep = True
            try:
                analysis.timestep_value = float(parts[1])
            except ValueError:
                pass

        # Thermo
        if cmd == "thermo":
            analysis.has_thermo = True

        # Run
        if cmd == "run" and len(parts) > 1:
            analysis.has_run_command = True
            try:
                analysis.total_run_steps += int(parts[1])
            except ValueError:
                pass

        # Output
        if cmd == "dump":
            analysis.has_dump = True
        if cmd == "write_restart" or cmd == "restart":
            analysis.has_restart = True
        if cmd == "write_data":
            analysis.has_write_data = True

    return analysis


@dataclass
class TestResult:
    """Complete test result including generation, validation, and analysis."""

    name: str
    success: bool
    deck_content: str
    analysis: DeckAnalysis
    validation_passed: bool
    validation_details: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    generation_time: float = 0.0


class TestMXeneEquilibration:
    """End-to-end tests for MXene equilibration deck generation."""

    @pytest.fixture
    def lammps_binary(self):
        """Get LAMMPS binary path if available."""
        # Check common locations
        candidates = [
            Path("/home/sf2/Workspace/main/39-GPUTests/1-GPUTests/md-lammps/install/bin/lmp"),
            Path("/usr/bin/lmp"),
            Path("/usr/local/bin/lmp"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_data_file_only(self, lammps_binary):
        """Test 1: Generate equilibration deck with only the data file."""
        intent = """Create a complete equilibration workflow for this MXene 2D material system.

The system is a Titanium Carbide MXene (Ti3C2) with OH surface terminations - a 2D ceramic material.

The equilibration should include:
1. Energy minimization to remove bad contacts
2. NPT equilibration at 298.15K and 1 atm to relax the lattice parameters (box dimensions)
3. NVT equilibration at 298.15K for production dynamics

Use appropriate settings for this type of system with the force field parameters already defined in the data file.
Output the complete input script that can be run directly.
"""

        reaper_input = ReaperInput(
            intent=intent,
            files=[DATA_FILE],
            lammps_binary=lammps_binary,
        )

        result = asyncio.run(generate_deck(reaper_input))

        # Analyze the deck
        analysis = analyze_deck(result.deck_content)

        print("\n" + "="*60)
        print("TEST 1: Data File Only")
        print("="*60)
        print(f"\nGeneration success: {result.success}")
        print(f"\nDeck Analysis:\n{analysis.summary()}")

        if result.validation:
            print(f"\nValidation:")
            print(f"  L0: {'✓' if result.validation.l0.passed else '✗'}")
            print(f"  L1: {'✓' if result.validation.l1.passed else '✗'}")
            print(f"  L2: {'✓' if result.validation.l2.passed else '✗'}")
            print(f"  L3: {'✓' if result.validation.l3.passed else '✗'}")

        print(f"\n--- Generated Deck ---")
        print(result.deck_content[:2000] + "..." if len(result.deck_content) > 2000 else result.deck_content)
        print("--- End Deck ---")

        # Store result for comparison
        self._test1_result = TestResult(
            name="Data File Only",
            success=result.success,
            deck_content=result.deck_content,
            analysis=analysis,
            validation_passed=result.validation.overall_passed if result.validation else False,
            validation_details={
                "L0": result.validation.l0.passed if result.validation else False,
                "L1": result.validation.l1.passed if result.validation else False,
                "L2": result.validation.l2.passed if result.validation else False,
                "L3": result.validation.l3.passed if result.validation else False,
            },
            errors=result.errors,
            warnings=result.warnings,
        )

        # Basic assertions
        assert result.deck_content, "Deck content should not be empty"
        assert analysis.has_units_real, "Should use real units for this system"
        assert analysis.has_read_data, "Should read the data file"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_data_file_with_context(self, lammps_binary):
        """Test 2: Generate equilibration deck with data file and context files."""
        intent = """Create a complete equilibration workflow for this MXene 2D material system.

The system is a Titanium Carbide MXene (Ti3C2) with OH surface terminations - a 2D ceramic material.

I've provided example input files showing the workflow structure:
- minimize_variable.inp: Shows how to set up minimization
- equilibration_npt_variable.inp: Shows NPT equilibration setup
- equilibration_nvt_variable.inp: Shows NVT equilibration setup
- run_equilibration_variable_data.sh: Shows the overall workflow

Please create a SINGLE combined input script that performs:
1. Energy minimization to remove bad contacts
2. NPT equilibration at 298.15K and 1 atm to relax the lattice parameters
3. NVT equilibration at 298.15K for production dynamics

The script should read the data file directly (not use variables) and be runnable as a single file.
"""

        context_files = [DATA_FILE, MINIMIZE_INP, NPT_INP, NVT_INP, WORKFLOW_SCRIPT]

        reaper_input = ReaperInput(
            intent=intent,
            files=context_files,
            lammps_binary=lammps_binary,
        )

        result = asyncio.run(generate_deck(reaper_input))

        # Analyze the deck
        analysis = analyze_deck(result.deck_content)

        print("\n" + "="*60)
        print("TEST 2: Data File + Context Files")
        print("="*60)
        print(f"\nGeneration success: {result.success}")
        print(f"\nDeck Analysis:\n{analysis.summary()}")

        if result.validation:
            print(f"\nValidation:")
            print(f"  L0: {'✓' if result.validation.l0.passed else '✗'}")
            print(f"  L1: {'✓' if result.validation.l1.passed else '✗'}")
            print(f"  L2: {'✓' if result.validation.l2.passed else '✗'}")
            print(f"  L3: {'✓' if result.validation.l3.passed else '✗'}")

        print(f"\n--- Generated Deck ---")
        print(result.deck_content[:2000] + "..." if len(result.deck_content) > 2000 else result.deck_content)
        print("--- End Deck ---")

        # Store result for comparison
        self._test2_result = TestResult(
            name="Data File + Context",
            success=result.success,
            deck_content=result.deck_content,
            analysis=analysis,
            validation_passed=result.validation.overall_passed if result.validation else False,
            validation_details={
                "L0": result.validation.l0.passed if result.validation else False,
                "L1": result.validation.l1.passed if result.validation else False,
                "L2": result.validation.l2.passed if result.validation else False,
                "L3": result.validation.l3.passed if result.validation else False,
            },
            errors=result.errors,
            warnings=result.warnings,
        )

        # Basic assertions
        assert result.deck_content, "Deck content should not be empty"
        assert analysis.has_units_real, "Should use real units for this system"
        assert analysis.has_read_data, "Should read the data file"


def run_comparison_test():
    """Run both tests and compare results."""
    import time

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    # Find LAMMPS binary
    lammps_binary = None
    for p in [
        Path("/home/sf2/Workspace/main/39-GPUTests/1-GPUTests/md-lammps/install/bin/lmp"),
        Path("/usr/bin/lmp"),
    ]:
        if p.exists():
            lammps_binary = p
            break

    print("="*70)
    print("MXene Equilibration Deck Generation Comparison")
    print("="*70)
    print(f"\nData file: {DATA_FILE}")
    print(f"Context files: {MINIMIZE_INP.name}, {NPT_INP.name}, {NVT_INP.name}")
    print(f"LAMMPS binary: {lammps_binary}")

    results = []

    # Test 1: Data file only
    print("\n" + "-"*70)
    print("Running Test 1: Data File Only")
    print("-"*70)

    intent1 = """Create a complete equilibration workflow for this MXene 2D material system.

The system is a Titanium Carbide MXene (Ti3C2) with OH surface terminations - a 2D ceramic material.

The equilibration should include:
1. Energy minimization to remove bad contacts
2. NPT equilibration at 298.15K and 1 atm to relax the lattice parameters (box dimensions)
3. NVT equilibration at 298.15K for production dynamics

Use appropriate settings for this type of system with the force field parameters already defined in the data file.
Output the complete input script that can be run directly.
"""

    start = time.time()
    result1 = asyncio.run(generate_deck(ReaperInput(
        intent=intent1,
        files=[DATA_FILE],
        lammps_binary=lammps_binary,
    )))
    time1 = time.time() - start

    analysis1 = analyze_deck(result1.deck_content)

    test1 = TestResult(
        name="Data File Only",
        success=result1.success,
        deck_content=result1.deck_content,
        analysis=analysis1,
        validation_passed=result1.validation.overall_passed if result1.validation else False,
        validation_details={
            "L0": result1.validation.l0.passed if result1.validation else False,
            "L1": result1.validation.l1.passed if result1.validation else False,
            "L2": result1.validation.l2.passed if result1.validation else False,
            "L3": result1.validation.l3.passed if result1.validation else False,
        },
        errors=result1.errors,
        warnings=result1.warnings,
        generation_time=time1,
    )
    results.append(test1)

    print(f"\nTest 1 completed in {time1:.1f}s")
    print(f"Score: {analysis1.score()}/100")

    # Test 2: Data file + context
    print("\n" + "-"*70)
    print("Running Test 2: Data File + Context Files")
    print("-"*70)

    intent2 = """Create a complete equilibration workflow for this MXene 2D material system.

The system is a Titanium Carbide MXene (Ti3C2) with OH surface terminations - a 2D ceramic material.

I've provided example input files showing the workflow structure:
- minimize_variable.inp: Shows how to set up minimization
- equilibration_npt_variable.inp: Shows NPT equilibration setup
- equilibration_nvt_variable.inp: Shows NVT equilibration setup
- run_equilibration_variable_data.sh: Shows the overall workflow

Please create a SINGLE combined input script that performs:
1. Energy minimization to remove bad contacts
2. NPT equilibration at 298.15K and 1 atm to relax the lattice parameters
3. NVT equilibration at 298.15K for production dynamics

The script should read the data file directly (not use variables) and be runnable as a single file.
"""

    start = time.time()
    result2 = asyncio.run(generate_deck(ReaperInput(
        intent=intent2,
        files=[DATA_FILE, MINIMIZE_INP, NPT_INP, NVT_INP, WORKFLOW_SCRIPT],
        lammps_binary=lammps_binary,
    )))
    time2 = time.time() - start

    analysis2 = analyze_deck(result2.deck_content)

    test2 = TestResult(
        name="Data File + Context",
        success=result2.success,
        deck_content=result2.deck_content,
        analysis=analysis2,
        validation_passed=result2.validation.overall_passed if result2.validation else False,
        validation_details={
            "L0": result2.validation.l0.passed if result2.validation else False,
            "L1": result2.validation.l1.passed if result2.validation else False,
            "L2": result2.validation.l2.passed if result2.validation else False,
            "L3": result2.validation.l3.passed if result2.validation else False,
        },
        errors=result2.errors,
        warnings=result2.warnings,
        generation_time=time2,
    )
    results.append(test2)

    print(f"\nTest 2 completed in {time2:.1f}s")
    print(f"Score: {analysis2.score()}/100")

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    print("\n{:<30} {:>15} {:>15}".format("Metric", "Test 1 (Data)", "Test 2 (+Context)"))
    print("-"*60)

    print("{:<30} {:>15} {:>15}".format(
        "Quality Score",
        f"{analysis1.score()}/100",
        f"{analysis2.score()}/100"
    ))

    print("{:<30} {:>15} {:>15}".format(
        "Validation Passed",
        "✓" if test1.validation_passed else "✗",
        "✓" if test2.validation_passed else "✗"
    ))

    print("{:<30} {:>15} {:>15}".format(
        "Generation Time",
        f"{test1.generation_time:.1f}s",
        f"{test2.generation_time:.1f}s"
    ))

    print("\n--- Essential Features ---")
    for feature, label in [
        ("has_units_real", "units real"),
        ("has_atom_style_full", "atom_style full"),
        ("has_read_data", "read_data"),
        ("has_pair_style_lj_coul", "pair_style lj/cut/coul"),
        ("has_kspace_pppm", "kspace pppm"),
    ]:
        v1 = "✓" if getattr(analysis1, feature) else "✗"
        v2 = "✓" if getattr(analysis2, feature) else "✗"
        print("{:<30} {:>15} {:>15}".format(label, v1, v2))

    print("\n--- Force Field ---")
    for feature, label in [
        ("has_bond_style_harmonic", "bond_style harmonic"),
        ("has_angle_style_harmonic", "angle_style harmonic"),
        ("has_dihedral_style", "dihedral_style"),
        ("has_improper_style", "improper_style"),
    ]:
        v1 = "✓" if getattr(analysis1, feature) else "✗"
        v2 = "✓" if getattr(analysis2, feature) else "✗"
        print("{:<30} {:>15} {:>15}".format(label, v1, v2))

    print("\n--- Simulation Phases ---")
    for feature, label in [
        ("has_minimize", "minimize"),
        ("has_npt_fix", "NPT fix"),
        ("has_nvt_fix", "NVT fix"),
    ]:
        v1 = "✓" if getattr(analysis1, feature) else "✗"
        v2 = "✓" if getattr(analysis2, feature) else "✗"
        print("{:<30} {:>15} {:>15}".format(label, v1, v2))

    print("\n--- Settings ---")
    print("{:<30} {:>15} {:>15}".format(
        "timestep",
        str(analysis1.timestep_value) if analysis1.timestep_value else "not set",
        str(analysis2.timestep_value) if analysis2.timestep_value else "not set"
    ))
    print("{:<30} {:>15} {:>15}".format(
        "temperature",
        str(analysis1.temperature) if analysis1.temperature else "not set",
        str(analysis2.temperature) if analysis2.temperature else "not set"
    ))
    print("{:<30} {:>15} {:>15}".format(
        "total run steps",
        str(analysis1.total_run_steps),
        str(analysis2.total_run_steps)
    ))

    print("\n--- Validation Details ---")
    for level in ["L0", "L1", "L2", "L3"]:
        v1 = "✓" if test1.validation_details.get(level) else "✗"
        v2 = "✓" if test2.validation_details.get(level) else "✗"
        print("{:<30} {:>15} {:>15}".format(level, v1, v2))

    # Score difference
    score_diff = analysis2.score() - analysis1.score()
    print("\n" + "="*70)
    print(f"SCORE DIFFERENCE: {score_diff:+d} points")
    if score_diff > 0:
        print("Context files IMPROVED the generated deck quality.")
    elif score_diff < 0:
        print("Data file only produced BETTER results (unexpected).")
    else:
        print("Both approaches produced EQUAL quality decks.")
    print("="*70)

    # Print full decks for review
    print("\n\n" + "="*70)
    print("FULL GENERATED DECKS")
    print("="*70)

    print("\n--- Test 1: Data File Only ---")
    print(result1.deck_content)

    print("\n--- Test 2: Data File + Context ---")
    print(result2.deck_content)

    return results


if __name__ == "__main__":
    run_comparison_test()
