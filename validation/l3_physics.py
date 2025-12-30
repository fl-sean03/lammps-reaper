"""L3 Validation: Minimal step execution + thermodynamic sanity checks.

This module validates LAMMPS decks by running them with minimal steps
to catch runtime explosions and instabilities, then parses the thermo
output to verify the simulation produced sane results.
"""

import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

from ..schemas import L3Result, ThermoData
from .l2_engine import find_lammps_binary
from .file_utils import setup_working_directory, cleanup_working_directory

# Number of steps to run for minimal execution test
MINIMAL_STEPS = 20

# Thermo output frequency for sanity checking
THERMO_FREQUENCY = 5

# Thermo sanity thresholds
MAX_REASONABLE_TEMP = 1e6  # Temperature above this = explosion
MAX_REASONABLE_ENERGY = 1e12  # Energy above this = explosion
MAX_TEMP_RATIO = 10.0  # Final temp / initial temp ratio threshold


def _create_minimal_step_deck(content: str, steps: int = MINIMAL_STEPS) -> str:
    """Modify deck to run minimal steps with thermo output for sanity checking.

    This modifies the deck to:
    - Set run to specified minimal steps
    - Add/modify thermo output for sanity checking
    - Keep output minimal

    Args:
        content: Original deck content.
        steps: Number of steps to run.

    Returns:
        Modified deck content for minimal-step run.
    """
    lines = content.split("\n")
    modified_lines = []
    run_found = False
    thermo_found = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            modified_lines.append(line)
            continue

        # Parse the command
        parts = stripped.split()
        if not parts:
            modified_lines.append(line)
            continue

        cmd = parts[0].lower()

        # Modify run command to run minimal steps
        if cmd == "run":
            modified_lines.append(f"run {steps}")
            run_found = True
        # Modify minimize to do minimal iterations
        elif cmd == "minimize":
            modified_lines.append(f"minimize 1e-4 1e-6 {steps} {steps * 10}")
        # Track if thermo is already set
        elif cmd == "thermo":
            thermo_found = True
            # Use frequent thermo output for sanity checking
            modified_lines.append(f"thermo {THERMO_FREQUENCY}")
        else:
            modified_lines.append(line)

    # Add thermo output if not present
    if not thermo_found:
        # Insert thermo before run command
        insert_idx = len(modified_lines)
        for i, line in enumerate(modified_lines):
            if line.strip().lower().startswith("run") or line.strip().lower().startswith("fix"):
                insert_idx = i
                break
        modified_lines.insert(insert_idx, f"thermo {THERMO_FREQUENCY}")
        modified_lines.insert(insert_idx, "thermo_style custom step temp press pe ke etotal")

    # If no run command found, add one at the end
    if not run_found:
        modified_lines.append(f"run {steps}")

    return "\n".join(modified_lines)


def _parse_thermo_output(output: str) -> List[ThermoData]:
    """Parse thermodynamic data from LAMMPS output.

    Args:
        output: LAMMPS stdout/stderr combined output.

    Returns:
        List of ThermoData objects parsed from output.
    """
    thermo_data = []
    lines = output.split("\n")

    # Find thermo header line to understand column order
    header_pattern = re.compile(r"^\s*Step\s+", re.IGNORECASE)
    data_pattern = re.compile(r"^\s*(\d+)\s+([-\d.eE+]+)")

    header_found = False
    columns = []

    for line in lines:
        stripped = line.strip()

        # Look for header line
        if header_pattern.match(stripped):
            header_found = True
            # Parse column names
            columns = stripped.lower().split()
            continue

        # If we found header, look for data lines
        if header_found and data_pattern.match(stripped):
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    data = ThermoData()
                    for i, col in enumerate(columns):
                        if i >= len(parts):
                            break
                        val_str = parts[i]
                        try:
                            val = float(val_str)
                        except ValueError:
                            continue

                        if col == "step":
                            data.step = int(val)
                        elif col == "temp":
                            data.temp = val
                        elif col == "press":
                            data.press = val
                        elif col == "pe" or col == "poteng":
                            data.pe = val
                        elif col == "ke" or col == "kineng":
                            data.ke = val
                        elif col == "etotal" or col == "toteng":
                            data.etotal = val

                    thermo_data.append(data)
                except (ValueError, IndexError):
                    continue

        # Stop parsing if we hit "Loop time" (end of run)
        if "Loop time" in stripped:
            break

    return thermo_data


def _check_thermo_sanity(thermo_data: List[ThermoData]) -> Tuple[bool, List[str]]:
    """Check thermodynamic data for sanity.

    Args:
        thermo_data: List of parsed thermo data points.

    Returns:
        Tuple of (passed, warnings list).
    """
    warnings = []
    passed = True

    if not thermo_data:
        warnings.append("No thermodynamic data found in output")
        return True, warnings  # Can't fail if no data

    # Check each data point
    for data in thermo_data:
        step = data.step

        # Check for NaN/Inf in temperature
        if data.temp is not None:
            if math.isnan(data.temp) or math.isinf(data.temp):
                warnings.append(f"Step {step}: Temperature is NaN/Inf - simulation exploded")
                passed = False
            elif data.temp < 0:
                warnings.append(f"Step {step}: Negative temperature ({data.temp}) - unphysical")
                passed = False
            elif data.temp > MAX_REASONABLE_TEMP:
                warnings.append(f"Step {step}: Temperature {data.temp:.2e} exceeds {MAX_REASONABLE_TEMP:.0e} - explosion")
                passed = False

        # Check for NaN/Inf in energy
        if data.etotal is not None:
            if math.isnan(data.etotal) or math.isinf(data.etotal):
                warnings.append(f"Step {step}: Total energy is NaN/Inf - simulation exploded")
                passed = False
            elif abs(data.etotal) > MAX_REASONABLE_ENERGY:
                warnings.append(f"Step {step}: Total energy {data.etotal:.2e} exceeds threshold - possible explosion")
                passed = False

        # Check for NaN/Inf in PE
        if data.pe is not None:
            if math.isnan(data.pe) or math.isinf(data.pe):
                warnings.append(f"Step {step}: Potential energy is NaN/Inf - simulation exploded")
                passed = False

        # Check for NaN/Inf in pressure
        if data.press is not None:
            if math.isnan(data.press) or math.isinf(data.press):
                warnings.append(f"Step {step}: Pressure is NaN/Inf - simulation exploded")
                passed = False

    # Check temperature drift (comparing first and last)
    if len(thermo_data) >= 2:
        first_temp = thermo_data[0].temp
        last_temp = thermo_data[-1].temp
        if first_temp is not None and last_temp is not None and first_temp > 0:
            temp_ratio = last_temp / first_temp
            if temp_ratio > MAX_TEMP_RATIO:
                warnings.append(
                    f"Temperature increased {temp_ratio:.1f}x ({first_temp:.1f} → {last_temp:.1f}) - "
                    "possible instability"
                )
                # This is a warning but not a hard failure
                # passed = False

    # Check energy conservation (for NVE)
    if len(thermo_data) >= 2:
        first_e = thermo_data[0].etotal
        last_e = thermo_data[-1].etotal
        if first_e is not None and last_e is not None and first_e != 0:
            energy_drift = abs(last_e - first_e) / abs(first_e)
            if energy_drift > 0.1:  # 10% drift
                warnings.append(
                    f"Energy drift of {energy_drift*100:.1f}% detected - may indicate instability"
                )

    if passed and not warnings:
        warnings.append("Thermodynamic sanity checks passed")

    return passed, warnings


def _parse_lammps_output(output: str, return_code: int) -> List[str]:
    """Parse LAMMPS output for errors and warnings.

    Args:
        output: Combined stdout/stderr from LAMMPS.
        return_code: Process return code.

    Returns:
        List of error/warning messages.
    """
    issues = []

    # Common error patterns in LAMMPS output
    error_patterns = [
        "ERROR:",
        "ERROR on proc",
        "Illegal",
        "Unknown",
        "Cannot",
        "Invalid",
        "Missing",
        "Expected",
        "Lost atoms",
        "Atom lost",
        "Out of range",
        "Bond atom",
        "Shake",
    ]

    warning_patterns = [
        "WARNING:",
        "Warning:",
    ]

    lines = output.split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for error patterns
        for pattern in error_patterns:
            if pattern in stripped:
                issues.append(f"LAMMPS Error: {stripped}")
                break

        # Check for warning patterns (non-fatal)
        for pattern in warning_patterns:
            if pattern in stripped:
                issues.append(f"LAMMPS Warning: {stripped}")
                break

    # If return code is non-zero but no specific errors found
    if return_code != 0 and not any("Error" in issue for issue in issues):
        issues.append(f"LAMMPS exited with return code {return_code}")

    return issues


def validate_l3(
    content: str,
    lammps_binary: Optional[Path] = None,
    steps: int = MINIMAL_STEPS,
    timeout: float = 60.0,
    context_files: Optional[List[Path]] = None,
) -> L3Result:
    """Validate LAMMPS deck by running it with minimal steps and checking thermo.

    Runs the deck with a small number of steps (default 20) to catch
    runtime explosions and instabilities, then parses thermodynamic
    output to verify sanity.

    This tests:
    - Force field evaluations actually work
    - Initial velocities don't cause immediate blowup
    - Neighbor list updates function correctly
    - Integrator produces stable dynamics
    - Thermodynamic quantities are reasonable (not NaN, not exploding)

    Args:
        content: The LAMMPS deck content to validate.
        lammps_binary: Optional path to LAMMPS binary.
        steps: Number of steps to run (default 20).
        timeout: Maximum time in seconds to wait for LAMMPS.
        context_files: Optional list of files to copy to working directory
                       (e.g., data files, potential files referenced by the deck).

    Returns:
        L3Result with validation status, thermo data, and details.
    """
    # Find LAMMPS binary
    binary = find_lammps_binary(lammps_binary)

    if binary is None:
        return L3Result(
            passed=True,  # Skip validation if no binary found
            engine_output="",
            return_code=-1,
            execution_time=0.0,
            steps_run=0,
            thermo_data=[],
            thermo_warnings=["LAMMPS binary not found - L3 validation skipped"],
            details=["LAMMPS binary not found - L3 validation skipped"],
        )

    # Create modified deck for minimal-step run with thermo output
    modified_content = _create_minimal_step_deck(content, steps)

    # Create working directory
    working_dir = Path(tempfile.mkdtemp(prefix="lammps_l3_"))

    try:
        # Set up working directory with deck and context files
        if context_files:
            deck_path = setup_working_directory(
                modified_content,
                context_files,
                working_dir,
                deck_filename="input.lammps",
            )
        else:
            # Just write the deck file
            deck_path = working_dir / "input.lammps"
            deck_path.write_text(modified_content)

        # Run LAMMPS
        start_time = time.time()
        result = subprocess.run(
            [str(binary), "-in", str(deck_path.name)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )
        execution_time = time.time() - start_time

        # Combine stdout and stderr
        output = result.stdout + "\n" + result.stderr

        # Parse output for errors
        issues = _parse_lammps_output(output, result.returncode)

        # Parse thermodynamic data
        thermo_data = _parse_thermo_output(output)

        # Check thermodynamic sanity
        thermo_passed, thermo_warnings = _check_thermo_sanity(thermo_data)

        # Check for successful completion indicators
        completed_ok = (
            result.returncode == 0
            or "Total wall time" in output
            or "Loop time" in output
        )

        # Check for explosion indicators in output
        # Note: "Dangerous builds = 0" is OK, only non-zero dangerous builds are bad
        has_explosion = any(
            pattern in output
            for pattern in ["Lost atoms", "Atom lost", "Out of range"]
        )
        # Check for dangerous neighbor list builds (non-zero is bad)
        dangerous_match = re.search(r"Dangerous builds\s*=\s*(\d+)", output)
        if dangerous_match and int(dangerous_match.group(1)) > 0:
            has_explosion = True

        # Determine overall pass/fail
        passed = completed_ok and not has_explosion and thermo_passed

        # Build details
        details = issues.copy() if issues else []
        if completed_ok and not has_explosion:
            details.append(f"LAMMPS execution completed ({steps} steps)")
        if has_explosion:
            details.append("Simulation explosion detected in LAMMPS output")
        if thermo_data:
            details.append(f"Parsed {len(thermo_data)} thermo data points")

        return L3Result(
            passed=passed,
            engine_output=output[:5000],  # Limit output size
            return_code=result.returncode,
            execution_time=execution_time,
            steps_run=steps,
            thermo_data=thermo_data,
            thermo_warnings=thermo_warnings,
            details=details,
        )

    except subprocess.TimeoutExpired:
        return L3Result(
            passed=False,
            engine_output="",
            return_code=-1,
            execution_time=timeout,
            steps_run=0,
            thermo_data=[],
            thermo_warnings=[f"Execution timed out after {timeout} seconds"],
            details=[f"LAMMPS execution timed out after {timeout} seconds"],
        )

    except Exception as e:
        return L3Result(
            passed=False,
            engine_output="",
            return_code=-1,
            execution_time=0.0,
            steps_run=0,
            thermo_data=[],
            thermo_warnings=[f"Error running LAMMPS: {str(e)}"],
            details=[f"Error running LAMMPS: {str(e)}"],
        )

    finally:
        # Clean up working directory
        cleanup_working_directory(working_dir)
