"""File utilities for LAMMPS validation.

This module provides utilities for handling file references in LAMMPS decks,
including parsing file references and copying context files to working directories.
"""

import re
import shutil
from pathlib import Path
from typing import List, Optional, Set


# LAMMPS commands that read external files
FILE_READING_COMMANDS = {
    "read_data": 0,       # read_data filename
    "read_restart": 0,    # read_restart filename
    "include": 0,         # include filename
    "read_dump": 0,       # read_dump filename ...
    "molecule": 1,        # molecule ID filename
    "pair_coeff": -1,     # pair_coeff ... filename (varies by pair_style)
    "bond_coeff": -1,     # may reference file
    "angle_coeff": -1,    # may reference file
}

# File extensions commonly used with LAMMPS
LAMMPS_FILE_EXTENSIONS = {
    ".data", ".dat",           # Data files
    ".restart", ".rst",        # Restart files
    ".lammps", ".in", ".inp",  # Input scripts
    ".eam", ".fs", ".alloy",   # EAM potentials
    ".tersoff", ".sw",         # Other potentials
    ".meam", ".comb", ".comb3",
    ".reax", ".ffield",
    ".txt", ".table",          # Table files
}


def parse_file_references(content: str) -> Set[str]:
    """Parse a LAMMPS deck and extract referenced file paths.

    Looks for commands like read_data, read_restart, include, molecule,
    and potential files in pair_coeff commands.

    Args:
        content: LAMMPS deck content.

    Returns:
        Set of file paths referenced in the deck.
    """
    referenced_files: Set[str] = set()
    lines = content.split("\n")

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        # Remove inline comments
        if "#" in stripped:
            stripped = stripped.split("#")[0].strip()

        parts = stripped.split()
        if not parts:
            continue

        cmd = parts[0].lower()

        # Handle specific commands
        if cmd == "read_data" and len(parts) > 1:
            referenced_files.add(parts[1])

        elif cmd == "read_restart" and len(parts) > 1:
            referenced_files.add(parts[1])

        elif cmd == "include" and len(parts) > 1:
            referenced_files.add(parts[1])

        elif cmd == "read_dump" and len(parts) > 1:
            referenced_files.add(parts[1])

        elif cmd == "molecule" and len(parts) > 2:
            referenced_files.add(parts[2])

        elif cmd == "pair_coeff":
            # Look for potential file references in pair_coeff
            # These usually have file extensions or are at the end
            for part in parts[1:]:
                # Skip wildcards and numbers
                if part in ("*", "NULL"):
                    continue
                try:
                    float(part)
                    continue
                except ValueError:
                    pass
                # Check if it looks like a filename
                if any(part.endswith(ext) for ext in LAMMPS_FILE_EXTENSIONS):
                    referenced_files.add(part)
                elif "/" in part or "." in part:
                    # Might be a path
                    if not part.startswith("-"):  # Not a flag
                        referenced_files.add(part)

        elif cmd in ("bond_coeff", "angle_coeff", "dihedral_coeff"):
            # Look for file references
            for part in parts[1:]:
                if any(part.endswith(ext) for ext in LAMMPS_FILE_EXTENSIONS):
                    referenced_files.add(part)

    return referenced_files


def find_file_in_context(
    filename: str,
    context_files: List[Path],
    search_dirs: Optional[List[Path]] = None,
) -> Optional[Path]:
    """Find a file in the context files or search directories.

    Args:
        filename: The filename to find (may be relative path).
        context_files: List of context file paths that were provided.
        search_dirs: Optional additional directories to search.

    Returns:
        Path to the file if found, None otherwise.
    """
    filename_path = Path(filename)
    filename_name = filename_path.name

    # Check if any context file matches the filename
    for cf in context_files:
        if cf.name == filename_name or str(cf).endswith(filename):
            if cf.exists():
                return cf

    # Build search directories from context files
    dirs_to_search: Set[Path] = set()
    for cf in context_files:
        if cf.exists():
            dirs_to_search.add(cf.parent)

    # Add explicit search directories
    if search_dirs:
        dirs_to_search.update(search_dirs)

    # Search in all directories
    for search_dir in dirs_to_search:
        # Try exact path
        candidate = search_dir / filename
        if candidate.exists():
            return candidate

        # Try just the filename
        candidate = search_dir / filename_name
        if candidate.exists():
            return candidate

    return None


def setup_working_directory(
    deck_content: str,
    context_files: List[Path],
    working_dir: Path,
    deck_filename: str = "input.lammps",
) -> Path:
    """Set up a working directory with all necessary files for LAMMPS.

    Copies the deck and all referenced files to the working directory.

    Args:
        deck_content: The LAMMPS deck content.
        context_files: List of context file paths to make available.
        working_dir: The working directory to set up.
        deck_filename: Name for the deck file in working directory.

    Returns:
        Path to the deck file in the working directory.
    """
    # Ensure working directory exists
    working_dir.mkdir(parents=True, exist_ok=True)

    # Write the deck
    deck_path = working_dir / deck_filename
    deck_path.write_text(deck_content)

    # Find all file references in the deck
    referenced_files = parse_file_references(deck_content)

    # Copy context files that are referenced
    copied_files: Set[str] = set()

    for ref_file in referenced_files:
        if ref_file in copied_files:
            continue

        # Find the file in context
        source_path = find_file_in_context(ref_file, context_files)

        if source_path and source_path.exists():
            # Determine destination path
            ref_path = Path(ref_file)
            if ref_path.is_absolute():
                dest_name = ref_path.name
            else:
                dest_name = ref_file

            dest_path = working_dir / dest_name

            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file
            if source_path != dest_path:
                shutil.copy2(source_path, dest_path)
                copied_files.add(ref_file)

    # Also copy any context files that might be needed but not explicitly referenced
    # (e.g., files that are referenced via variables or includes)
    for cf in context_files:
        if cf.exists() and cf.is_file():
            dest_path = working_dir / cf.name
            if not dest_path.exists():
                shutil.copy2(cf, dest_path)

    return deck_path


def cleanup_working_directory(
    working_dir: Path,
    keep_outputs: bool = False,
) -> None:
    """Clean up a working directory after validation.

    Args:
        working_dir: The working directory to clean up.
        keep_outputs: If True, keep output files for debugging.
    """
    if not working_dir.exists():
        return

    if keep_outputs:
        # Only remove input files, keep outputs
        patterns_to_remove = ["*.lammps", "input.*"]
        for pattern in patterns_to_remove:
            for f in working_dir.glob(pattern):
                try:
                    f.unlink()
                except OSError:
                    pass
    else:
        # Remove entire directory
        try:
            shutil.rmtree(working_dir)
        except OSError:
            pass
