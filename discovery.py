"""
File discovery module for LAMMPS Reaper.

Automatically discovers and categorizes LAMMPS-related files in a directory.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# File extension mappings
DATA_EXTENSIONS = {".data", ".dat"}
INPUT_EXTENSIONS = {".in", ".lmp", ".lammps", ".inp"}
POTENTIAL_EXTENSIONS = {
    ".eam", ".fs", ".alloy",  # EAM potentials
    ".tersoff",  # Tersoff
    ".sw",  # Stillinger-Weber
    ".meam", ".library",  # MEAM
    ".reax", ".ffield",  # ReaxFF
    ".comb", ".comb3",  # COMB
    ".table", ".txt",  # Table potentials
}
RESTART_EXTENSIONS = {".restart", ".rst"}


@dataclass
class DiscoveredFiles:
    """Container for discovered LAMMPS files in a directory."""

    directory: Path
    data_files: list[Path] = field(default_factory=list)
    input_files: list[Path] = field(default_factory=list)
    potential_files: list[Path] = field(default_factory=list)
    restart_files: list[Path] = field(default_factory=list)
    other_files: list[Path] = field(default_factory=list)

    @property
    def primary_data_file(self) -> Optional[Path]:
        """Returns the primary data file (first .data file found)."""
        return self.data_files[0] if self.data_files else None

    @property
    def context_files(self) -> list[Path]:
        """Returns all files that can serve as context (inputs + potentials)."""
        return self.input_files + self.potential_files

    @property
    def all_files(self) -> list[Path]:
        """Returns all discovered files."""
        return (
            self.data_files +
            self.input_files +
            self.potential_files +
            self.restart_files +
            self.other_files
        )

    def summary(self) -> str:
        """Returns a human-readable summary of discovered files."""
        lines = [f"Directory: {self.directory}"]

        if self.data_files:
            lines.append(f"  Data files ({len(self.data_files)}):")
            for f in self.data_files:
                lines.append(f"    - {f.name}")

        if self.input_files:
            lines.append(f"  Input scripts ({len(self.input_files)}):")
            for f in self.input_files:
                lines.append(f"    - {f.name}")

        if self.potential_files:
            lines.append(f"  Potential files ({len(self.potential_files)}):")
            for f in self.potential_files:
                lines.append(f"    - {f.name}")

        if self.restart_files:
            lines.append(f"  Restart files ({len(self.restart_files)}):")
            for f in self.restart_files:
                lines.append(f"    - {f.name}")

        if not self.all_files:
            lines.append("  No LAMMPS files found")

        return "\n".join(lines)


def classify_file(path: Path) -> str:
    """
    Classify a file based on its extension.

    Returns one of: 'data', 'input', 'potential', 'restart', 'other'
    """
    suffix = path.suffix.lower()
    name_lower = path.name.lower()

    # Check compound extensions like .eam.fs, .eam.alloy
    if ".eam." in name_lower:
        return "potential"

    if suffix in DATA_EXTENSIONS:
        return "data"
    elif suffix in INPUT_EXTENSIONS:
        return "input"
    elif suffix in POTENTIAL_EXTENSIONS:
        return "potential"
    elif suffix in RESTART_EXTENSIONS:
        return "restart"
    else:
        return "other"


def discover_files(
    directory: Path,
    recursive: bool = False,
    include_hidden: bool = False,
) -> DiscoveredFiles:
    """
    Discover and categorize LAMMPS files in a directory.

    Args:
        directory: Path to the directory to scan
        recursive: If True, scan subdirectories as well
        include_hidden: If True, include hidden files (starting with .)

    Returns:
        DiscoveredFiles object with categorized files

    Raises:
        ValueError: If directory doesn't exist or isn't a directory
    """
    directory = Path(directory).resolve()

    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    result = DiscoveredFiles(directory=directory)

    # Get files
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for path in directory.glob(pattern):
        # Skip directories
        if path.is_dir():
            continue

        # Skip hidden files unless requested
        if not include_hidden and path.name.startswith("."):
            continue

        # Classify and store
        file_type = classify_file(path)

        if file_type == "data":
            result.data_files.append(path)
        elif file_type == "input":
            result.input_files.append(path)
        elif file_type == "potential":
            result.potential_files.append(path)
        elif file_type == "restart":
            result.restart_files.append(path)
        # Skip 'other' files - we only care about LAMMPS files

    # Sort all lists by name for consistency
    result.data_files.sort(key=lambda p: p.name)
    result.input_files.sort(key=lambda p: p.name)
    result.potential_files.sort(key=lambda p: p.name)
    result.restart_files.sort(key=lambda p: p.name)
    result.other_files.sort(key=lambda p: p.name)

    return result


def generate_output_filename(directory: Path, prefix: str = "generated") -> Path:
    """
    Generate a unique output filename for the generated LAMMPS script.

    Args:
        directory: Directory where the file will be created
        prefix: Prefix for the filename

    Returns:
        Path to the output file (doesn't check if it exists)
    """
    base_name = f"{prefix}.in"
    output_path = directory / base_name

    # If file exists, add a number
    counter = 1
    while output_path.exists():
        output_path = directory / f"{prefix}_{counter}.in"
        counter += 1

    return output_path
