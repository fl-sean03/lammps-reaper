"""L1 Validation: Syntax + Physics parameter checks.

This module validates LAMMPS command syntax and physics parameters without
running the engine. Combines structural syntax checks with physics sanity checks.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from ..schemas import L1Result

# Timestep ranges for different unit systems (min, max) in native units
TIMESTEP_RANGES: Dict[str, Tuple[float, float]] = {
    "lj": (0.0001, 0.01),  # LJ reduced units
    "real": (0.1, 10.0),  # femtoseconds
    "metal": (0.0001, 0.01),  # picoseconds
    "si": (1e-18, 1e-14),  # seconds
    "cgs": (1e-18, 1e-14),  # seconds
    "electron": (0.0001, 0.01),  # femtoseconds
    "micro": (0.1, 100.0),  # microseconds
    "nano": (0.0001, 1.0),  # nanoseconds
}

# Temperature ranges for warnings (in Kelvin or reduced units)
TEMPERATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "lj": (0.01, 10.0),  # Reduced units
    "real": (1.0, 10000.0),  # Kelvin
    "metal": (1.0, 10000.0),  # Kelvin
    "si": (1.0, 10000.0),  # Kelvin
    "cgs": (1.0, 10000.0),  # Kelvin
    "electron": (1.0, 10000.0),  # Kelvin
    "micro": (1.0, 10000.0),  # Kelvin
    "nano": (1.0, 10000.0),  # Kelvin
}

# Required commands that should be present in most LAMMPS scripts
REQUIRED_COMMANDS = {
    "units": "No 'units' command found - required for defining unit system",
    "atom_style": "No 'atom_style' command found - required for defining atom properties",
}

# Commands for structure definition (at least one should be present)
STRUCTURE_COMMANDS = {"read_data", "read_restart", "create_box", "create_atoms"}

# Commands that require pair definitions
PAIR_COMMANDS = {"pair_style", "pair_coeff"}

# Common LAMMPS commands for validation
KNOWN_COMMANDS = {
    "units",
    "atom_style",
    "dimension",
    "boundary",
    "newton",
    "processors",
    "read_data",
    "read_restart",
    "read_dump",
    "create_box",
    "create_atoms",
    "lattice",
    "region",
    "group",
    "mass",
    "pair_style",
    "pair_coeff",
    "pair_modify",
    "bond_style",
    "bond_coeff",
    "angle_style",
    "angle_coeff",
    "dihedral_style",
    "dihedral_coeff",
    "improper_style",
    "improper_coeff",
    "kspace_style",
    "neighbor",
    "neigh_modify",
    "fix",
    "unfix",
    "compute",
    "uncompute",
    "variable",
    "thermo",
    "thermo_style",
    "thermo_modify",
    "dump",
    "undump",
    "dump_modify",
    "restart",
    "minimize",
    "min_style",
    "min_modify",
    "run",
    "velocity",
    "timestep",
    "reset_timestep",
    "set",
    "change_box",
    "replicate",
    "displace_atoms",
    "delete_atoms",
    "write_data",
    "write_restart",
    "write_dump",
    "info",
    "print",
    "shell",
    "label",
    "jump",
    "if",
    "next",
    "loop",
    "include",
    "clear",
    "log",
    "echo",
    "partition",
    "python",
    "special_bonds",
    "dielectric",
    "suffix",
    "package",
    "comm_style",
    "comm_modify",
}

# Valid unit systems
VALID_UNITS = {"lj", "real", "metal", "si", "cgs", "electron", "micro", "nano"}

# Valid atom styles
VALID_ATOM_STYLES = {
    "atomic",
    "angle",
    "bond",
    "charge",
    "dipole",
    "dpd",
    "edpd",
    "mdpd",
    "tdpd",
    "electron",
    "ellipsoid",
    "full",
    "line",
    "meso",
    "molecular",
    "peri",
    "smd",
    "sphere",
    "spin",
    "template",
    "tri",
    "wavepacket",
    "hybrid",
}


def _parse_command(line: str) -> Optional[Tuple[str, List[str]]]:
    """Parse a LAMMPS command line into command and arguments.

    Args:
        line: A single line from the LAMMPS deck.

    Returns:
        Tuple of (command, arguments) or None if not a command.
    """
    # Remove comments
    line = line.split("#")[0].strip()

    if not line:
        return None

    # Handle line continuations (lines ending with &)
    # For simplicity, we just process the current line

    parts = line.split()
    if not parts:
        return None

    command = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []

    return command, args


def _check_required_commands(
    commands_found: Set[str],
) -> List[Tuple[str, str]]:
    """Check for required commands.

    Args:
        commands_found: Set of commands found in the deck.

    Returns:
        List of (error_type, message) tuples.
    """
    errors = []

    for cmd, message in REQUIRED_COMMANDS.items():
        if cmd not in commands_found:
            errors.append(("missing_required", message))

    return errors


def _check_structure_commands(
    commands_found: Set[str],
) -> List[Tuple[str, str]]:
    """Check for structure definition commands.

    Args:
        commands_found: Set of commands found in the deck.

    Returns:
        List of (error_type, message) tuples.
    """
    errors = []

    has_structure = any(cmd in commands_found for cmd in STRUCTURE_COMMANDS)
    if not has_structure:
        errors.append(
            (
                "missing_structure",
                "No structure command found - need one of: "
                + ", ".join(sorted(STRUCTURE_COMMANDS)),
            )
        )

    return errors


def _check_pair_definitions(
    commands_found: Set[str],
) -> List[Tuple[str, str]]:
    """Check for pair style/coeff definitions.

    Args:
        commands_found: Set of commands found in the deck.

    Returns:
        List of (warning_type, message) tuples.
    """
    warnings = []

    has_pair_style = "pair_style" in commands_found
    has_pair_coeff = "pair_coeff" in commands_found

    if has_pair_style and not has_pair_coeff:
        warnings.append(
            ("missing_pair_coeff", "pair_style defined but no pair_coeff found")
        )
    elif has_pair_coeff and not has_pair_style:
        warnings.append(
            ("missing_pair_style", "pair_coeff defined but no pair_style found")
        )

    return warnings


def _validate_units_command(args: List[str]) -> Optional[str]:
    """Validate units command arguments.

    Args:
        args: Arguments to the units command.

    Returns:
        Error message if invalid, None otherwise.
    """
    if not args:
        return "units command requires a unit system argument"

    unit_system = args[0].lower()
    if unit_system not in VALID_UNITS:
        return f"Unknown unit system '{unit_system}' - valid options: {', '.join(sorted(VALID_UNITS))}"

    return None


def _validate_atom_style_command(args: List[str]) -> Optional[str]:
    """Validate atom_style command arguments.

    Args:
        args: Arguments to the atom_style command.

    Returns:
        Error message if invalid, None otherwise.
    """
    if not args:
        return "atom_style command requires a style argument"

    style = args[0].lower()
    if style not in VALID_ATOM_STYLES:
        return f"Unknown atom style '{style}' - check LAMMPS documentation for valid styles"

    return None


def _check_common_syntax_errors(
    content: str,
) -> List[Tuple[int, str]]:
    """Check for common syntax errors.

    Args:
        content: The LAMMPS deck content.

    Returns:
        List of (line_number, error_message) tuples.
    """
    errors = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        # Check for unbalanced quotes
        single_quotes = stripped.count("'")
        double_quotes = stripped.count('"')
        if single_quotes % 2 != 0:
            errors.append((line_num, "Unbalanced single quotes"))
        if double_quotes % 2 != 0:
            errors.append((line_num, "Unbalanced double quotes"))

        # Check for unbalanced parentheses
        open_parens = stripped.count("(")
        close_parens = stripped.count(")")
        if open_parens != close_parens:
            errors.append((line_num, "Unbalanced parentheses"))

        # Check for unbalanced brackets
        open_brackets = stripped.count("[")
        close_brackets = stripped.count("]")
        if open_brackets != close_brackets:
            errors.append((line_num, "Unbalanced brackets"))

        # Check for unbalanced braces
        open_braces = stripped.count("{")
        close_braces = stripped.count("}")
        if open_braces != close_braces:
            errors.append((line_num, "Unbalanced braces"))

        # Check for multiple commands on same line (without proper separation)
        # LAMMPS uses newlines to separate commands
        parsed = _parse_command(stripped)
        if parsed:
            cmd, args = parsed
            # Check if command is known
            if cmd not in KNOWN_COMMANDS and not cmd.startswith("fix_") and not cmd.startswith("compute_"):
                # This might be a typo or unknown command
                # We'll issue a warning, not an error
                pass

    return errors


# =============================================================================
# Physics Parameter Validation Functions
# =============================================================================


def _parse_timestep(content: str) -> Optional[float]:
    """Parse the timestep command from the deck."""
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.match(r"timestep\s+([\d.eE+-]+)", stripped, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _parse_temperatures(content: str) -> List[Tuple[float, int, str]]:
    """Parse temperature values from the deck."""
    temperatures = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Check for fix nvt/npt temperature
        if re.search(r"\b(nvt|npt|langevin|temp/berendsen)\b", stripped, re.IGNORECASE):
            temp_match = re.search(
                r"temp\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
                stripped,
                re.IGNORECASE,
            )
            if temp_match:
                try:
                    t_start = float(temp_match.group(1))
                    t_end = float(temp_match.group(2))
                    temperatures.append((t_start, line_num, "fix temperature start"))
                    temperatures.append((t_end, line_num, "fix temperature end"))
                except ValueError:
                    pass

        # Check for velocity command temperature
        if stripped.lower().startswith("velocity"):
            vel_match = re.search(
                r"velocity\s+\S+\s+create\s+([\d.eE+-]+)",
                stripped,
                re.IGNORECASE,
            )
            if vel_match:
                try:
                    temp = float(vel_match.group(1))
                    temperatures.append((temp, line_num, "velocity create"))
                except ValueError:
                    pass

    return temperatures


def _check_timestep_physics(
    timestep: float,
    units: str,
) -> List[str]:
    """Check if timestep is within reasonable range."""
    warnings = []

    if units not in TIMESTEP_RANGES:
        return warnings

    min_ts, max_ts = TIMESTEP_RANGES[units]

    if timestep < min_ts:
        warnings.append(
            f"Timestep {timestep} is very small for '{units}' units "
            f"(recommended: {min_ts} - {max_ts}). This may cause slow simulations."
        )
    elif timestep > max_ts:
        warnings.append(
            f"Timestep {timestep} is very large for '{units}' units "
            f"(recommended: {min_ts} - {max_ts}). This may cause instability."
        )

    return warnings


def _check_temperature_physics(
    temperatures: List[Tuple[float, int, str]],
    units: str,
) -> List[str]:
    """Check if temperatures are within reasonable range."""
    warnings = []

    if units not in TEMPERATURE_RANGES:
        return warnings

    min_temp, max_temp = TEMPERATURE_RANGES[units]

    for temp, line_num, context in temperatures:
        if temp < 0:
            warnings.append(
                f"Line {line_num}: Negative temperature {temp} in {context}. "
                "Temperature must be positive."
            )
        elif temp < min_temp:
            warnings.append(
                f"Line {line_num}: Very low temperature {temp} in {context} "
                f"(typical range: {min_temp} - {max_temp})."
            )
        elif temp > max_temp:
            warnings.append(
                f"Line {line_num}: Very high temperature {temp} in {context} "
                f"(typical range: {min_temp} - {max_temp})."
            )

    return warnings


def validate_l1(content: str) -> L1Result:
    """Validate LAMMPS command syntax and physics parameters.

    Checks for:
    - Required commands (units, atom_style)
    - Structure commands (read_data, create_box, etc.)
    - Pair style/coeff definitions
    - Common syntax errors (unbalanced quotes, parentheses, etc.)
    - Physics parameters (timestep, temperature ranges)

    Args:
        content: The LAMMPS deck content to validate.

    Returns:
        L1Result with validation status and details.
    """
    syntax_errors: List[str] = []
    physics_warnings: List[str] = []
    line_numbers: List[int] = []
    details: List[str] = []
    commands_found: Set[str] = set()
    command_args: Dict[str, List[str]] = {}

    # Parse all commands
    lines = content.split("\n")
    for line_num, line in enumerate(lines, start=1):
        parsed = _parse_command(line)
        if parsed:
            cmd, args = parsed
            commands_found.add(cmd)
            command_args[cmd] = args

    # Check for required commands
    for error_type, message in _check_required_commands(commands_found):
        syntax_errors.append(message)
        details.append(f"Error: {message}")

    # Check for structure commands
    for error_type, message in _check_structure_commands(commands_found):
        syntax_errors.append(message)
        details.append(f"Error: {message}")

    # Check for pair definitions
    for warning_type, message in _check_pair_definitions(commands_found):
        details.append(f"Warning: {message}")

    # Validate specific command arguments
    if "units" in command_args:
        error = _validate_units_command(command_args["units"])
        if error:
            syntax_errors.append(error)
            details.append(f"Error: {error}")

    if "atom_style" in command_args:
        error = _validate_atom_style_command(command_args["atom_style"])
        if error:
            syntax_errors.append(error)
            details.append(f"Error: {error}")

    # Check for common syntax errors
    for line_num, error in _check_common_syntax_errors(content):
        syntax_errors.append(f"Line {line_num}: {error}")
        line_numbers.append(line_num)
        details.append(f"Syntax error at line {line_num}: {error}")

    # ==========================================================================
    # Physics Parameter Validation
    # ==========================================================================

    # Determine unit system (default to 'lj' if not specified)
    units = "lj"
    if "units" in command_args and command_args["units"]:
        units = command_args["units"][0].lower()

    # Check timestep
    timestep = _parse_timestep(content)
    if timestep is not None:
        ts_warnings = _check_timestep_physics(timestep, units)
        physics_warnings.extend(ts_warnings)
        for w in ts_warnings:
            details.append(f"Physics warning: {w}")
    else:
        details.append("No timestep command found, using LAMMPS default")

    # Check temperatures
    temperatures = _parse_temperatures(content)
    if temperatures:
        temp_warnings = _check_temperature_physics(temperatures, units)
        physics_warnings.extend(temp_warnings)
        for w in temp_warnings:
            details.append(f"Physics warning: {w}")

    # Critical physics errors that should fail validation (e.g., negative temperature)
    critical_physics = [w for w in physics_warnings if "Negative temperature" in w]

    # Validation passes if no syntax errors and no critical physics errors
    passed = len(syntax_errors) == 0 and len(critical_physics) == 0

    return L1Result(
        passed=passed,
        syntax_errors=syntax_errors,
        physics_warnings=physics_warnings,
        line_numbers=line_numbers,
        details=details,
    )
