#!/usr/bin/env python3
"""Command-line interface for LAMMPS Reaper.

LAMMPS Reaper is an AI-powered LAMMPS input deck generator that uses
LLM technology to create production-ready molecular dynamics simulation
scripts from natural language descriptions.

Usage:
    lammps-reaper generate "Run NVT simulation at 300K" ./my_project/
    lammps-reaper validate ./my_project/
    lammps-reaper analyze ./my_project/
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from . import (
    __version__,
    ReaperInput,
    generate_deck,
    validate_deck,
    validate_l0,
    validate_l1,
    validate_l2,
    validate_l3,
    find_lammps_binary,
    analyze_data_file,
    build_file_context,
    detect_file_type,
)
from .discovery import discover_files, generate_output_filename


def print_banner():
    """Print the LAMMPS Reaper banner."""
    banner = f"""
╔═══════════════════════════════════════════════════════════════╗
║                      LAMMPS REAPER v{__version__:<24}║
║         AI-Powered LAMMPS Input Deck Generator                ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def cmd_generate(args):
    """Generate a LAMMPS input deck from natural language intent."""
    print_banner()

    # Resolve directory
    directory = Path(args.directory).resolve()
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return 1
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        return 1

    # Discover files
    print(f"Scanning directory: {directory}")
    discovered = discover_files(directory)
    print(discovered.summary())
    print()

    # Build file list: data files first, then context (input scripts + potentials)
    files = []
    if discovered.data_files:
        files.extend(discovered.data_files)
    if discovered.input_files:
        files.extend(discovered.input_files)
    if discovered.potential_files:
        files.extend(discovered.potential_files)

    if not files:
        print("Warning: No LAMMPS files found in directory")
        print("Proceeding with generation without file context...")

    # Find LAMMPS binary
    lammps_binary = None
    if args.lammps:
        lammps_binary = Path(args.lammps)
        if not lammps_binary.exists():
            print(f"Warning: LAMMPS binary not found: {args.lammps}")
            lammps_binary = None
    else:
        lammps_binary = find_lammps_binary()

    # Set output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = directory / output_path
    else:
        output_path = generate_output_filename(directory)

    # Build input
    reaper_input = ReaperInput(
        intent=args.intent,
        files=files,
        output_path=output_path,
        lammps_binary=lammps_binary,
        max_retries=args.max_retries,
        enable_iterative_fixing=not args.no_fix,
    )

    print(f"Intent: {args.intent[:100]}{'...' if len(args.intent) > 100 else ''}")
    print(f"Files discovered: {len(files)}")
    if lammps_binary:
        print(f"LAMMPS binary: {lammps_binary}")
    print(f"Output: {output_path}")
    print(f"Iterative fixing: {'enabled' if not args.no_fix else 'disabled'}")
    print()
    print("Generating LAMMPS deck...")
    print("-" * 60)

    # Run generation
    try:
        result = asyncio.run(generate_deck(reaper_input))
    except KeyboardInterrupt:
        print("\nGeneration cancelled.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Print results
    print()
    print("=" * 60)
    print("GENERATION RESULTS")
    print("=" * 60)

    print(f"\nSuccess: {'Yes' if result.success else 'No'}")
    print(f"Total attempts: {result.total_attempts}")

    # Print assumptions
    if result.assumptions and args.show_assumptions:
        print(f"\n--- Assumptions Made ({len(result.assumptions)}) ---")
        for a in result.assumptions:
            print(f"  [{a.category.value}] {a.description}")
            if args.verbose:
                print(f"    Value: {a.assumed_value}")
                print(f"    Reasoning: {a.reasoning}")
                print(f"    Confidence: {a.confidence}")

    # Print validation results
    if result.validation:
        print("\n--- Validation ---")
        v = result.validation
        print(f"  L0 (Placeholders): {'PASS' if v.l0.passed else 'FAIL'}")
        print(f"  L1 (Syntax/Physics): {'PASS' if v.l1.passed else 'FAIL'}")
        print(f"  L2 (Engine): {'PASS' if v.l2.passed else 'FAIL'} ({v.l2.execution_time:.1f}s)")
        print(f"  L3 (Execution): {'PASS' if v.l3.passed else 'FAIL'} ({v.l3.execution_time:.1f}s)")
        print(f"  Overall: {'PASS' if v.overall_passed else 'FAIL'}")

        if not v.overall_passed and args.verbose:
            print("\n  Issues:")
            for issue in v.issues:
                print(f"    - {issue}")

    # Print errors/warnings
    if result.errors:
        print("\n--- Errors ---")
        for e in result.errors:
            print(f"  - {e}")

    if result.warnings:
        print("\n--- Warnings ---")
        for w in result.warnings:
            print(f"  - {w}")

    # Output
    if result.success:
        print(f"\nOutput written to: {output_path}")

    if args.print_deck or args.verbose:
        print("\n" + "=" * 60)
        print("GENERATED DECK")
        print("=" * 60)
        print(result.deck_content)

    return 0 if result.success else 1


def cmd_validate(args):
    """Validate a LAMMPS input deck or directory."""
    print_banner()

    target = Path(args.target).resolve()

    # Find LAMMPS binary
    lammps_binary = None
    if args.lammps:
        lammps_binary = Path(args.lammps)
    else:
        lammps_binary = find_lammps_binary()

    if target.is_dir():
        # Directory mode: validate all input files in directory
        discovered = discover_files(target)
        print(f"Scanning directory: {target}")
        print(discovered.summary())
        print()

        if not discovered.input_files:
            print("No LAMMPS input files found to validate.")
            return 1

        # Use all other files as context
        context_files = discovered.data_files + discovered.potential_files

        overall_success = True
        for deck_path in discovered.input_files:
            print(f"\n{'=' * 60}")
            print(f"Validating: {deck_path.name}")
            print("=" * 60)

            content = deck_path.read_text()
            success = _validate_content(content, lammps_binary, context_files, args)
            if not success:
                overall_success = False

        return 0 if overall_success else 1

    elif target.is_file():
        # Single file mode
        content = target.read_text()
        print(f"Validating: {target}")

        # Check for context in same directory
        context_files = []
        discovered = discover_files(target.parent)
        context_files = discovered.data_files + discovered.potential_files
        if context_files:
            print(f"Context files found: {len(context_files)}")

        if lammps_binary:
            print(f"LAMMPS binary: {lammps_binary}")
        print()

        success = _validate_content(content, lammps_binary, context_files, args)
        return 0 if success else 1

    else:
        print(f"Error: Not found: {target}")
        return 1


def _validate_content(content, lammps_binary, context_files, args):
    """Validate content and print results. Returns True if passed."""
    # Run specific level or all
    if args.level:
        level = args.level.upper()
        if level == "L0":
            result = validate_l0(content)
            print(f"L0 (Placeholders): {'PASS' if result.passed else 'FAIL'}")
            if not result.passed:
                print(f"  Unresolved: {result.unresolved_count}")
                for p in result.placeholders_found:
                    print(f"    - {p}")
            return result.passed

        elif level == "L1":
            result = validate_l1(content)
            print(f"L1 (Syntax/Physics): {'PASS' if result.passed else 'FAIL'}")
            if result.syntax_errors:
                print("  Syntax errors:")
                for e in result.syntax_errors:
                    print(f"    - {e}")
            if result.physics_warnings:
                print("  Physics warnings:")
                for w in result.physics_warnings:
                    print(f"    - {w}")
            return result.passed

        elif level == "L2":
            result = validate_l2(content, lammps_binary, context_files=context_files or None)
            print(f"L2 (Engine): {'PASS' if result.passed else 'FAIL'}")
            print(f"  Execution time: {result.execution_time:.2f}s")
            if not result.passed and args.verbose:
                print(f"  Output: {result.engine_output[:500]}")
            return result.passed

        elif level == "L3":
            result = validate_l3(content, lammps_binary, context_files=context_files or None)
            print(f"L3 (Execution): {'PASS' if result.passed else 'FAIL'}")
            print(f"  Steps run: {result.steps_run}")
            print(f"  Execution time: {result.execution_time:.2f}s")
            if result.thermo_warnings:
                print("  Thermo warnings:")
                for w in result.thermo_warnings:
                    print(f"    - {w}")
            return result.passed

        else:
            print(f"Unknown level: {level}. Use L0, L1, L2, or L3.")
            return False
    else:
        # Run all levels
        result = validate_deck(content, lammps_binary, context_files=context_files or None)

        print(f"\nL0 (Placeholders): {'PASS' if result.l0.passed else 'FAIL'}")
        if not result.l0.passed:
            print(f"  Unresolved: {result.l0.unresolved_count}")

        print(f"\nL1 (Syntax/Physics): {'PASS' if result.l1.passed else 'FAIL'}")
        if result.l1.syntax_errors:
            for e in result.l1.syntax_errors[:5]:
                print(f"  - {e}")
        if result.l1.physics_warnings:
            for w in result.l1.physics_warnings[:5]:
                print(f"  - {w}")

        print(f"\nL2 (Engine): {'PASS' if result.l2.passed else 'FAIL'}")
        print(f"  Execution time: {result.l2.execution_time:.2f}s")

        print(f"\nL3 (Execution): {'PASS' if result.l3.passed else 'FAIL'}")
        print(f"  Steps run: {result.l3.steps_run}")
        print(f"  Execution time: {result.l3.execution_time:.2f}s")

        print()
        print("=" * 40)
        print(f"OVERALL: {'PASS' if result.overall_passed else 'FAIL'}")
        print("=" * 40)

        if result.issues:
            print("\nIssues found:")
            for issue in result.issues:
                print(f"  - {issue}")

        return result.overall_passed


def cmd_analyze(args):
    """Analyze a LAMMPS file or directory."""
    print_banner()

    target = Path(args.target).resolve()

    if target.is_dir():
        # Directory mode
        discovered = discover_files(target)
        print(discovered.summary())
        print()

        # Analyze each file
        all_files = discovered.data_files + discovered.input_files + discovered.potential_files

        if not all_files:
            print("No LAMMPS files found to analyze.")
            return 0

        analyses = {}
        for file_path in all_files:
            print(f"\n{'=' * 60}")
            analysis = _analyze_file(file_path, args.verbose)
            if analysis:
                analyses[str(file_path.name)] = analysis

        if args.json:
            print("\n" + "-" * 60)
            print("JSON output:")
            print(json.dumps(analyses, indent=2))

    elif target.is_file():
        # Single file mode
        analysis = _analyze_file(target, args.verbose)

        if args.json and analysis:
            print("\n" + "-" * 60)
            print("JSON output:")
            print(json.dumps(analysis, indent=2))

    else:
        print(f"Error: Not found: {target}")
        return 1

    return 0


def _analyze_file(file_path, verbose=False):
    """Analyze a single file and print results. Returns analysis dict or None."""
    content = file_path.read_text()
    file_type = detect_file_type(file_path, content)

    print(f"File: {file_path.name}")
    print(f"Type: {file_type}")

    if file_type == "data_file":
        analysis = analyze_data_file(content)

        print(f"\nTopology:")
        print(f"  Atom types: {analysis['atom_types']}")
        print(f"  Has bonds: {'Yes' if analysis['has_bonds'] else 'No'} ({analysis['bond_types']} types)")
        print(f"  Has angles: {'Yes' if analysis['has_angles'] else 'No'} ({analysis['angle_types']} types)")
        print(f"  Has dihedrals: {'Yes' if analysis['has_dihedrals'] else 'No'} ({analysis['dihedral_types']} types)")
        print(f"  Has impropers: {'Yes' if analysis['has_impropers'] else 'No'} ({analysis['improper_types']} types)")
        print(f"  Has charges: {'Yes' if analysis['has_charges'] else 'No'}")

        if analysis['units_hint']:
            print(f"\nUnits hint: {analysis['units_hint']}")

        print("\nRequired LAMMPS style declarations:")
        if analysis['has_bonds']:
            print("  - bond_style (e.g., harmonic)")
        if analysis['has_angles']:
            print("  - angle_style (e.g., harmonic)")
        if analysis['has_dihedrals']:
            print("  - dihedral_style (e.g., harmonic)")
        if analysis['has_impropers']:
            print("  - improper_style (e.g., cvff)")
        if analysis['has_charges']:
            print("  - kspace_style (e.g., pppm 1.0e-5)")

        return analysis

    elif file_type == "input_file":
        lines = content.split("\n")
        commands = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                parts = stripped.split()
                if parts:
                    commands.append(parts[0].lower())

        analysis = {
            "file_type": "input_file",
            "lines": len(lines),
            "commands": len(commands),
        }

        print(f"\nLines: {len(lines)}")
        print(f"Commands: {len(commands)}")

        # Check for key commands
        key_commands = ["units", "atom_style", "pair_style", "read_data",
                       "fix", "run", "minimize", "dump"]
        found_commands = {}
        for cmd in key_commands:
            count = commands.count(cmd)
            if count > 0:
                found_commands[cmd] = count

        if found_commands:
            print("\nKey commands found:")
            for cmd, count in found_commands.items():
                print(f"  - {cmd}: {count}")
            analysis["key_commands"] = found_commands

        return analysis

    else:
        print(f"  No detailed analysis available for type '{file_type}'")
        return {"file_type": file_type}


def cmd_info(args):
    """Show information about LAMMPS Reaper and environment."""
    print_banner()

    print("Environment:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Package location: {Path(__file__).parent}")

    # Check for LAMMPS binary
    lammps_binary = find_lammps_binary()
    print(f"\nLAMMPS binary: {lammps_binary if lammps_binary else 'Not found'}")

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    print(f"ANTHROPIC_API_KEY: {'Set' if api_key else 'Not set'}")

    print("\nValidation Levels:")
    print("  L0: Placeholder detection ({{VAR}}, <VAR>, TODO, FIXME)")
    print("  L1: LAMMPS syntax and physics parameter validation")
    print("  L2: LAMMPS engine acceptance (run 0 steps)")
    print("  L3: Minimal execution test (~20 steps)")

    print("\nSupported file types:")
    print("  - .data, .dat: LAMMPS data files")
    print("  - .in, .lmp, .lammps, .inp: LAMMPS input scripts")
    print("  - .eam, .tersoff, .sw, .meam: Potential files")

    print("\nUsage:")
    print("  lammps-reaper generate \"your intent\" ./directory/")
    print("  lammps-reaper validate ./directory/")
    print("  lammps-reaper analyze ./directory/")

    return 0


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="lammps-reaper",
        description="LAMMPS Reaper - AI-Powered LAMMPS Input Deck Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a deck - just point at your project directory
  lammps-reaper generate "Run NVT equilibration at 300K" ./my_simulation/

  # Validate all input files in a directory
  lammps-reaper validate ./my_simulation/

  # Analyze all LAMMPS files in a directory
  lammps-reaper analyze ./my_simulation/

  # Show environment info
  lammps-reaper info

The CLI automatically discovers:
  - .data, .dat files as system data files
  - .in, .inp, .lammps files as context/examples
  - .eam, .tersoff, .sw, .meam files as potentials
        """,
    )

    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"lammps-reaper {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate a LAMMPS input deck from natural language",
        aliases=["gen", "g"],
    )
    gen_parser.add_argument(
        "intent",
        help="Natural language description of desired simulation",
    )
    gen_parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing LAMMPS files (default: current directory)",
    )
    gen_parser.add_argument(
        "-o", "--output",
        help="Output filename (default: generated.in in the target directory)",
    )
    gen_parser.add_argument(
        "-l", "--lammps",
        help="Path to LAMMPS binary for validation",
    )
    gen_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for iterative fixing (default: 3)",
    )
    gen_parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Disable iterative fixing on validation failure",
    )
    gen_parser.add_argument(
        "--show-assumptions",
        action="store_true",
        default=True,
        help="Show assumptions made by the LLM (default: True)",
    )
    gen_parser.add_argument(
        "--print-deck",
        action="store_true",
        help="Print the generated deck to stdout",
    )
    gen_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    gen_parser.set_defaults(func=cmd_generate)

    # Validate command
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate LAMMPS input file(s)",
        aliases=["val", "v"],
    )
    val_parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Directory or file to validate (default: current directory)",
    )
    val_parser.add_argument(
        "-l", "--level",
        choices=["L0", "L1", "L2", "L3", "l0", "l1", "l2", "l3"],
        help="Run only a specific validation level",
    )
    val_parser.add_argument(
        "--lammps",
        help="Path to LAMMPS binary",
    )
    val_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    val_parser.set_defaults(func=cmd_validate)

    # Analyze command
    ana_parser = subparsers.add_parser(
        "analyze",
        help="Analyze LAMMPS file(s)",
        aliases=["ana", "a"],
    )
    ana_parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Directory or file to analyze (default: current directory)",
    )
    ana_parser.add_argument(
        "--json",
        action="store_true",
        help="Output analysis as JSON",
    )
    ana_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    ana_parser.set_defaults(func=cmd_analyze)

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show environment and configuration info",
    )
    info_parser.set_defaults(func=cmd_info)

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Run command
    return args.func(args)


def main_sync():
    """Synchronous entry point."""
    sys.exit(main())


if __name__ == "__main__":
    main_sync()
