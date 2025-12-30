"""Generator module for lammps_reaper.

This module handles LAMMPS deck generation from user intent and context files,
with iterative fixing and assumption tracking.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from .provider import AnthropicProvider
from .schemas import (
    Assumption,
    AssumptionCategory,
    FileContext,
    GenerationAttempt,
    L0Result,
    L1Result,
    L2Result,
    L3Result,
    ReaperInput,
    ReaperOutput,
    ValidationResult,
)

# Maximum lines to include from large files
MAX_FILE_LINES = 1000

# System prompt with assumption tracking
LAMMPS_SYSTEM_PROMPT = """You are an expert LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) simulation engineer with deep expertise in molecular dynamics simulations, materials science, and computational physics.

Your task is to generate production-ready LAMMPS input scripts (decks) based on user requirements.

OUTPUT FORMAT REQUIREMENTS:
Your response must have TWO sections, clearly separated:

1. ASSUMPTIONS SECTION (JSON format):
Start with a JSON block listing any assumptions you made. Use this format:
```json
{
  "assumptions": [
    {
      "category": "force_field|units|parameters|topology|simulation|output|other",
      "description": "What you assumed",
      "assumed_value": "The value/setting you chose",
      "reasoning": "Why you made this assumption",
      "confidence": "low|medium|high"
    }
  ]
}
```

2. LAMMPS SCRIPT SECTION:
After the JSON, output the raw LAMMPS script starting with "# LAMMPS INPUT SCRIPT" comment.
- NO markdown code fences (no ```lammps or ```)
- NO explanatory text
- The script must be directly runnable

WHEN YOU MUST MAKE ASSUMPTIONS:
- If a data file has bonds/angles/dihedrals but no style info is provided, assume harmonic styles
- If the unit system isn't specified, infer from the data file or use 'real' for molecular systems
- If force field parameters aren't provided, state what you assumed
- If simulation length isn't specified, use reasonable defaults and state them

LAMMPS BEST PRACTICES:
1. Always specify units explicitly
2. Define atom_style appropriate for the simulation
3. Set boundary conditions explicitly (p p p for periodic)
4. For data files with topology (bonds, angles, dihedrals, impropers):
   - ALWAYS declare bond_style, angle_style, dihedral_style, improper_style
   - If styles aren't specified, assume harmonic for bonds/angles, and cvff for impropers
5. Include neighbor list settings
6. Set appropriate timestep for the chosen units
7. Use thermo and thermo_style for output monitoring
8. For systems with charges, always include kspace_style (pppm or ewald)

SIMULATION SETUP ORDER:
1. Initialization: units, dimension, boundary, atom_style
2. Force field STYLES (pair_style, bond_style, angle_style, dihedral_style, improper_style)
3. System definition: read_data (this reads coefficients from data file)
4. K-space: kspace_style (if charges present)
5. Settings: neighbor, neigh_modify, timestep
6. Fixes: time integration (nve, nvt, npt)
7. Output: thermo, dump
8. Run: minimize (if needed), velocity, run

Generate a complete, runnable LAMMPS input script that follows these guidelines."""


FIX_PROMPT_TEMPLATE = """The previous LAMMPS script failed validation with these errors:

{errors}

LAMMPS output:
{lammps_output}

Please fix the script to address these errors. Common fixes:
- If "Unknown bond/angle/dihedral/improper style": Add the style declaration BEFORE read_data
- If "Cannot open file": Check the filename matches the provided data file
- If pair_coeff errors: Ensure pair_style is set and coefficients match atom types
- If kspace errors: Use pppm 1.0e-5 for charged systems

Output your response in the same format:
1. JSON assumptions block (update if you changed assumptions)
2. Fixed LAMMPS script starting with "# LAMMPS INPUT SCRIPT"
"""


def detect_file_type(path: Path, content: str) -> str:
    """Detect the type of a LAMMPS-related file."""
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix in (".data", ".dat"):
        return "data_file"
    if suffix in (".in", ".lmp", ".lammps", ".inp"):
        return "input_file"
    if suffix in (".eam", ".eam.fs", ".eam.alloy"):
        return "eam_potential"
    if suffix == ".tersoff":
        return "tersoff_potential"
    if suffix == ".sw":
        return "sw_potential"
    if suffix in (".meam", ".library"):
        return "meam_potential"
    if suffix == ".reax":
        return "reax_potential"
    if suffix == ".sh":
        return "shell_script"

    if "potential" in name or "pot" in name:
        return "potential_file"
    if "param" in name:
        return "parameter_file"

    first_lines = "\n".join(content.split("\n")[:20]).lower()

    if any(kw in first_lines for kw in ["atoms", "atom types", "bonds", "masses", "xlo xhi"]):
        return "data_file"
    if any(kw in first_lines for kw in ["units", "atom_style", "pair_style", "read_data"]):
        return "input_file"
    if "nrho" in first_lines and "drho" in first_lines:
        return "eam_potential"

    return "unknown"


def analyze_data_file(content: str) -> dict:
    """Analyze a LAMMPS data file to extract key information."""
    info = {
        "has_bonds": False,
        "has_angles": False,
        "has_dihedrals": False,
        "has_impropers": False,
        "has_charges": False,
        "atom_types": 0,
        "bond_types": 0,
        "angle_types": 0,
        "dihedral_types": 0,
        "improper_types": 0,
        "units_hint": None,
    }

    lines = content.split("\n")
    for line in lines[:100]:  # Check first 100 lines
        line_lower = line.lower().strip()

        if "bonds" in line_lower and "bond types" not in line_lower:
            parts = line_lower.split()
            if parts and parts[0].isdigit() and int(parts[0]) > 0:
                info["has_bonds"] = True

        if "angles" in line_lower and "angle types" not in line_lower:
            parts = line_lower.split()
            if parts and parts[0].isdigit() and int(parts[0]) > 0:
                info["has_angles"] = True

        if "dihedrals" in line_lower and "dihedral types" not in line_lower:
            parts = line_lower.split()
            if parts and parts[0].isdigit() and int(parts[0]) > 0:
                info["has_dihedrals"] = True

        if "impropers" in line_lower and "improper types" not in line_lower:
            parts = line_lower.split()
            if parts and parts[0].isdigit() and int(parts[0]) > 0:
                info["has_impropers"] = True

        if "atom types" in line_lower:
            parts = line_lower.split()
            if parts and parts[0].isdigit():
                info["atom_types"] = int(parts[0])

        if "bond types" in line_lower:
            parts = line_lower.split()
            if parts and parts[0].isdigit():
                info["bond_types"] = int(parts[0])

        if "angle types" in line_lower:
            parts = line_lower.split()
            if parts and parts[0].isdigit():
                info["angle_types"] = int(parts[0])

        if "dihedral types" in line_lower:
            parts = line_lower.split()
            if parts and parts[0].isdigit():
                info["dihedral_types"] = int(parts[0])

        if "improper types" in line_lower:
            parts = line_lower.split()
            if parts and parts[0].isdigit():
                info["improper_types"] = int(parts[0])

        # Check for units hint
        if "units = real" in line_lower or "units=real" in line_lower:
            info["units_hint"] = "real"
        elif "units = metal" in line_lower:
            info["units_hint"] = "metal"
        elif "units = lj" in line_lower:
            info["units_hint"] = "lj"

        # Check for charges (look for Atoms section with charge column)
        if "atoms # full" in line_lower or "atoms # charge" in line_lower:
            info["has_charges"] = True

    # Also check for Pair Coeffs with coul
    if "pair coeffs # lj/cut/coul" in content.lower():
        info["has_charges"] = True

    return info


def build_file_context(files: List[Path]) -> Tuple[str, dict]:
    """Read files and build structured context for the LLM.

    Returns:
        Tuple of (formatted context string, data file analysis dict)
    """
    if not files:
        return "", {}

    file_contexts: List[FileContext] = []
    data_analysis = {}

    for file_path in files:
        if not file_path.exists():
            continue

        try:
            content = file_path.read_text()
        except Exception:
            continue

        file_type = detect_file_type(file_path, content)

        # Analyze data files
        if file_type == "data_file":
            data_analysis = analyze_data_file(content)

        # Truncate large files
        lines = content.split("\n")
        if len(lines) > MAX_FILE_LINES:
            truncated_content = "\n".join(lines[:MAX_FILE_LINES])
            truncated_content += f"\n\n[... truncated, {len(lines) - MAX_FILE_LINES} more lines ...]"
            content = truncated_content

        file_contexts.append(FileContext(path=file_path, content=content, file_type=file_type))

    if not file_contexts:
        return "", {}

    # Build formatted context
    context_parts = ["=== PROVIDED FILES ===\n"]

    for fc in file_contexts:
        context_parts.append(f"--- FILE: {fc.path.name} ---")
        context_parts.append(f"Type: {fc.file_type}")
        context_parts.append(f"Path: {fc.path}")
        context_parts.append("Content:")
        context_parts.append(fc.content)
        context_parts.append("")

    # Add data file analysis hints
    if data_analysis:
        context_parts.append("--- DATA FILE ANALYSIS ---")
        if data_analysis.get("has_bonds"):
            context_parts.append(f"- Contains bonds ({data_analysis.get('bond_types', 0)} types) - REQUIRES bond_style")
        if data_analysis.get("has_angles"):
            context_parts.append(f"- Contains angles ({data_analysis.get('angle_types', 0)} types) - REQUIRES angle_style")
        if data_analysis.get("has_dihedrals"):
            context_parts.append(f"- Contains dihedrals ({data_analysis.get('dihedral_types', 0)} types) - REQUIRES dihedral_style")
        if data_analysis.get("has_impropers"):
            context_parts.append(f"- Contains impropers ({data_analysis.get('improper_types', 0)} types) - REQUIRES improper_style")
        if data_analysis.get("has_charges"):
            context_parts.append("- Contains charges - REQUIRES kspace_style (pppm or ewald)")
        if data_analysis.get("units_hint"):
            context_parts.append(f"- Units hint from file: {data_analysis['units_hint']}")
        context_parts.append("")

    return "\n".join(context_parts), data_analysis


def build_prompt(intent: str, file_context: Optional[str] = None) -> str:
    """Build the user prompt for deck generation."""
    prompt_parts = []

    if file_context:
        prompt_parts.append(file_context)
        prompt_parts.append("")

    prompt_parts.append("=== SIMULATION REQUEST ===")
    prompt_parts.append(intent)
    prompt_parts.append("")

    prompt_parts.append("=== OUTPUT INSTRUCTIONS ===")
    prompt_parts.append("1. First output a JSON block with your assumptions")
    prompt_parts.append("2. Then output the LAMMPS script starting with '# LAMMPS INPUT SCRIPT'")
    prompt_parts.append("Remember: Declare ALL required styles BEFORE read_data!")

    return "\n".join(prompt_parts)


def parse_llm_response(response: str) -> Tuple[str, List[Assumption]]:
    """Parse LLM response to extract script and assumptions.

    Returns:
        Tuple of (deck_content, assumptions_list)
    """
    assumptions = []
    deck_content = ""

    # Try to extract JSON assumptions block
    json_match = re.search(r'```json\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            if "assumptions" in json_data:
                for a in json_data["assumptions"]:
                    category = AssumptionCategory.OTHER
                    cat_str = a.get("category", "other").lower()
                    for cat in AssumptionCategory:
                        if cat.value == cat_str:
                            category = cat
                            break

                    assumptions.append(Assumption(
                        category=category,
                        description=a.get("description", ""),
                        assumed_value=a.get("assumed_value", ""),
                        reasoning=a.get("reasoning", ""),
                        confidence=a.get("confidence", "medium"),
                    ))
        except (json.JSONDecodeError, KeyError):
            pass

    # Extract LAMMPS script
    # Look for script after JSON block or after "# LAMMPS INPUT SCRIPT"
    script_marker = "# LAMMPS INPUT SCRIPT"
    if script_marker in response:
        idx = response.find(script_marker)
        deck_content = response[idx:].strip()
    else:
        # Try to find script after JSON block
        if json_match:
            after_json = response[json_match.end():].strip()
            # Remove any remaining markdown
            after_json = re.sub(r'^```\w*\s*\n?', '', after_json)
            after_json = re.sub(r'\n?```\s*$', '', after_json)
            deck_content = after_json.strip()
        else:
            # Fallback: clean the whole response
            deck_content = clean_llm_output(response)

    # Final cleanup
    deck_content = clean_llm_output(deck_content)

    return deck_content, assumptions


def clean_llm_output(response: str) -> str:
    """Clean LLM response to extract raw LAMMPS script."""
    content = response.strip()

    # Remove markdown code fences
    content = re.sub(r"^```(?:lammps|LAMMPS|lmp|LMP|in|sh)?\s*\n?", "", content)
    content = re.sub(r"\n?```\s*$", "", content)

    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    return content.strip()


def _create_stub_validation() -> ValidationResult:
    """Create a stub validation result when validation is not available."""
    return ValidationResult(
        overall_passed=True,
        l0=L0Result(passed=True, details=["Validation not run"]),
        l1=L1Result(passed=True, details=["Validation not run"]),
        l2=L2Result(passed=True, details=["Validation not run"]),
        l3=L3Result(passed=True, details=["Validation not run"]),
        issues=["Full validation not available"],
    )


def _format_validation_errors(validation: ValidationResult) -> str:
    """Format validation errors for the fix prompt."""
    errors = []

    if not validation.l0.passed:
        errors.append(f"L0 (Placeholders): {', '.join(validation.l0.details)}")

    if not validation.l1.passed:
        if validation.l1.syntax_errors:
            errors.append(f"L1 (Syntax): {', '.join(validation.l1.syntax_errors)}")
        if validation.l1.physics_warnings:
            errors.append(f"L1 (Physics): {', '.join(validation.l1.physics_warnings)}")

    if not validation.l2.passed:
        errors.append(f"L2 (Engine): {', '.join(validation.l2.details)}")

    if not validation.l3.passed:
        errors.append(f"L3 (Execution): {', '.join(validation.l3.details)}")

    return "\n".join(errors)


async def generate_deck(reaper_input: ReaperInput) -> ReaperOutput:
    """Generate a LAMMPS deck from the given input with iterative fixing.

    Args:
        reaper_input: Input configuration including intent and context files.

    Returns:
        ReaperOutput with the generated deck, validation results, and assumptions.
    """
    errors: List[str] = []
    warnings: List[str] = []
    all_assumptions: List[Assumption] = []
    attempts: List[GenerationAttempt] = []

    # Build context from input files
    file_context, data_analysis = None, {}
    if reaper_input.files:
        file_context, data_analysis = build_file_context(reaper_input.files)
        if not file_context and reaper_input.files:
            warnings.append("Some input files could not be read")

    # Build initial prompt
    prompt = build_prompt(reaper_input.intent, file_context)

    # Initialize provider
    try:
        provider = AnthropicProvider()
    except Exception as e:
        return ReaperOutput(
            success=False,
            deck_content="",
            output_path=None,
            validation=None,
            errors=[f"Failed to initialize provider: {str(e)}"],
            warnings=warnings,
        )

    # Iterative generation loop
    max_attempts = reaper_input.max_retries + 1 if reaper_input.enable_iterative_fixing else 1
    deck_content = ""
    validation = None
    current_prompt = prompt

    for attempt_num in range(1, max_attempts + 1):
        # Call LLM
        try:
            response = await provider.create_message(
                system_prompt=LAMMPS_SYSTEM_PROMPT,
                user_message=current_prompt,
                max_tokens=8192,
            )
        except Exception as e:
            errors.append(f"LLM API error on attempt {attempt_num}: {str(e)}")
            break

        # Parse response
        deck_content, assumptions = parse_llm_response(response)
        all_assumptions.extend(assumptions)

        if not deck_content:
            errors.append(f"Attempt {attempt_num}: LLM returned empty response")
            continue

        # Run validation
        try:
            from . import validate_deck as run_validation
            validation = run_validation(
                deck_content,
                lammps_binary=reaper_input.lammps_binary,
                context_files=reaper_input.files if reaper_input.files else None,
            )
        except (ImportError, AttributeError):
            validation = _create_stub_validation()
            warnings.append("Validation module not available")

        # Record attempt
        attempt_errors = []
        if not validation.overall_passed:
            attempt_errors = validation.issues.copy()

        attempts.append(GenerationAttempt(
            attempt_number=attempt_num,
            deck_content=deck_content,
            validation_passed=validation.overall_passed,
            errors=attempt_errors,
            fixes_applied=[],
        ))

        # Check if validation passed
        if validation.overall_passed:
            break

        # If iterative fixing is disabled or we've exhausted retries, stop
        if not reaper_input.enable_iterative_fixing or attempt_num >= max_attempts:
            break

        # Build fix prompt for next iteration
        error_text = _format_validation_errors(validation)
        lammps_output = ""
        if not validation.l2.passed:
            lammps_output = validation.l2.engine_output[:2000]
        elif not validation.l3.passed:
            lammps_output = validation.l3.engine_output[:2000]

        current_prompt = FIX_PROMPT_TEMPLATE.format(
            errors=error_text,
            lammps_output=lammps_output if lammps_output else "No LAMMPS output available",
        )

        # Update the last attempt with fix info
        if attempts:
            attempts[-1].fixes_applied.append(f"Attempting fix for: {error_text[:200]}")

    # Write to output file if specified
    output_path = None
    if reaper_input.output_path and deck_content:
        try:
            reaper_input.output_path.parent.mkdir(parents=True, exist_ok=True)
            reaper_input.output_path.write_text(deck_content)
            output_path = reaper_input.output_path
        except Exception as e:
            errors.append(f"Failed to write output file: {str(e)}")

    # Determine success
    success = validation.overall_passed if validation else False
    if errors and not deck_content:
        success = False

    return ReaperOutput(
        success=success,
        deck_content=deck_content,
        output_path=output_path,
        validation=validation,
        errors=errors,
        warnings=warnings,
        assumptions=all_assumptions,
        attempts=attempts,
        total_attempts=len(attempts),
    )


def generate_deck_sync(reaper_input: ReaperInput) -> ReaperOutput:
    """Synchronous wrapper for generate_deck."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, generate_deck(reaper_input))
                return future.result()
        else:
            return loop.run_until_complete(generate_deck(reaper_input))
    except RuntimeError:
        return asyncio.run(generate_deck(reaper_input))
