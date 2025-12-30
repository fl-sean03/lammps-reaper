"""L2 Validation: Engine execution checks.

This module validates LAMMPS decks by running them through the LAMMPS engine.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from ..schemas import L2Result
from .file_utils import setup_working_directory, cleanup_working_directory

# Default paths to check for LAMMPS binary
LAMMPS_PATHS = [
    Path("/home/sf2/Workspace/main/39-GPUTests/1-GPUTests/md-lammps/install/bin/lmp"),
    Path("/usr/local/bin/lmp"),
    Path("/usr/bin/lmp"),
    Path("/usr/local/bin/lmp_serial"),
    Path("/usr/bin/lmp_serial"),
    Path("/usr/local/bin/lmp_mpi"),
    Path("/usr/bin/lmp_mpi"),
    Path("/opt/lammps/bin/lmp"),
    Path.home() / "lammps" / "build" / "lmp",
    Path.home() / "lammps" / "src" / "lmp_serial",
]


def find_lammps_binary(user_path: Optional[Path] = None) -> Optional[Path]:
    """Find a valid LAMMPS binary.

    Searches in the following order:
    1. User-provided path
    2. LAMMPS_BINARY environment variable
    3. Common installation paths

    Args:
        user_path: Optional user-specified path to LAMMPS binary.

    Returns:
        Path to LAMMPS binary if found, None otherwise.
    """
    # Check user-provided path first
    if user_path is not None:
        if user_path.exists() and user_path.is_file():
            return user_path

    # Check environment variable
    env_path = os.environ.get("LAMMPS_BINARY")
    if env_path:
        env_path_obj = Path(env_path)
        if env_path_obj.exists() and env_path_obj.is_file():
            return env_path_obj

    # Check common paths
    for path in LAMMPS_PATHS:
        if path.exists() and path.is_file():
            return path

    return None


def _create_zero_step_deck(content: str) -> str:
    """Modify deck to run zero steps for syntax checking.

    This modifies the deck to:
    - Set run to 0 steps
    - Disable unnecessary output

    Args:
        content: Original deck content.

    Returns:
        Modified deck content for zero-step run.
    """
    lines = content.split("\n")
    modified_lines = []
    run_found = False

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

        # Modify run command to run 0 steps
        if cmd == "run":
            modified_lines.append("run 0")
            run_found = True
        # Modify minimize to do 0 iterations
        elif cmd == "minimize":
            modified_lines.append("minimize 0 0 0 0")
        else:
            modified_lines.append(line)

    # If no run command found, add one at the end
    if not run_found:
        modified_lines.append("run 0")

    return "\n".join(modified_lines)


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

        # Check for warning patterns
        for pattern in warning_patterns:
            if pattern in stripped:
                issues.append(f"LAMMPS Warning: {stripped}")
                break

    # If return code is non-zero but no specific errors found
    if return_code != 0 and not any("Error" in issue for issue in issues):
        issues.append(f"LAMMPS exited with return code {return_code}")

    return issues


def validate_l2(
    content: str,
    lammps_binary: Optional[Path] = None,
    timeout: float = 30.0,
    context_files: Optional[List[Path]] = None,
) -> L2Result:
    """Validate LAMMPS deck by running it through the engine.

    Runs the deck with zero steps to check for syntax and initialization errors.

    Args:
        content: The LAMMPS deck content to validate.
        lammps_binary: Optional path to LAMMPS binary.
        timeout: Maximum time in seconds to wait for LAMMPS.
        context_files: Optional list of files to copy to working directory
                       (e.g., data files, potential files referenced by the deck).

    Returns:
        L2Result with validation status and details.
    """
    # Find LAMMPS binary
    binary = find_lammps_binary(lammps_binary)

    if binary is None:
        return L2Result(
            passed=True,  # Skip validation if no binary found
            engine_output="",
            return_code=-1,
            execution_time=0.0,
            details=["LAMMPS binary not found - L2 validation skipped"],
        )

    # Create modified deck for zero-step run
    modified_content = _create_zero_step_deck(content)

    # Create working directory
    working_dir = Path(tempfile.mkdtemp(prefix="lammps_l2_"))

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

        # Determine pass/fail
        passed = result.returncode == 0 and not any("Error" in issue for issue in issues)

        return L2Result(
            passed=passed,
            engine_output=output[:5000],  # Limit output size
            return_code=result.returncode,
            execution_time=execution_time,
            details=issues if issues else ["LAMMPS validation passed"],
        )

    except subprocess.TimeoutExpired:
        return L2Result(
            passed=False,
            engine_output="",
            return_code=-1,
            execution_time=timeout,
            details=[f"LAMMPS execution timed out after {timeout} seconds"],
        )

    except Exception as e:
        return L2Result(
            passed=False,
            engine_output="",
            return_code=-1,
            execution_time=0.0,
            details=[f"Error running LAMMPS: {str(e)}"],
        )

    finally:
        # Clean up working directory
        cleanup_working_directory(working_dir)
