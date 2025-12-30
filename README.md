# LAMMPS Reaper

**AI-Powered LAMMPS Input Deck Generator with Multi-Level Validation**

LAMMPS Reaper uses large language models (LLMs) to generate production-ready LAMMPS molecular dynamics simulation scripts from natural language descriptions. It features automatic validation, iterative error correction, and transparent assumption tracking.

## Features

- **Natural Language Generation**: Describe your simulation in plain English, get a runnable LAMMPS script
- **Directory-Based Workflow**: Point at a directory, Reaper discovers all your files automatically
- **Multi-Level Validation (L0-L3)**: Comprehensive validation from placeholder detection to actual execution
- **Iterative Error Fixing**: Automatically retries with error feedback when validation fails
- **Assumption Tracking**: LLM explicitly states all assumptions made during generation
- **Data File Analysis**: Automatically detects topology (bonds, angles, charges) and required styles
- **Context-Aware**: Uses your existing input scripts as style guides

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/fl-sean03/lammps-reaper.git
cd lammps-reaper

# Install in development mode
pip install -e ".[dev]"

# Or install dependencies directly
pip install anthropic python-dotenv
```

### Setup

1. **Get an Anthropic API Key**: Sign up at [console.anthropic.com](https://console.anthropic.com)

2. **Set the environment variable**:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or create a `.env` file in your working directory:
```
ANTHROPIC_API_KEY=your-api-key-here
```

3. **(Optional) Set LAMMPS binary path** for L2/L3 validation:
```bash
export LAMMPS_BINARY="/path/to/lmp"
```

### Verify Setup

```bash
lammps-reaper info
```

## Usage

### The Simple Way: Point at a Directory

LAMMPS Reaper automatically discovers all relevant files in your project directory:

```bash
# Generate a LAMMPS deck - just point at your project directory
lammps-reaper generate "Run NVT equilibration at 300K" ./my_simulation/

# Validate all input files in a directory
lammps-reaper validate ./my_simulation/

# Analyze all LAMMPS files in a directory
lammps-reaper analyze ./my_simulation/
```

**What gets discovered automatically:**
- `.data`, `.dat` → System data files (used for topology analysis)
- `.in`, `.inp`, `.lammps`, `.lmp` → Input scripts (used as context/examples)
- `.eam`, `.tersoff`, `.sw`, `.meam` → Potential files

### Command Line Interface

#### Generate a LAMMPS Deck

```bash
# Basic generation - reads files from current directory
lammps-reaper generate "Create a Lennard-Jones fluid NVT simulation at T=1.0"

# Point at a specific project directory
lammps-reaper generate "Equilibrate this polymer system at 300K" ./polymer_project/

# Specify output filename
lammps-reaper generate "Run NPT equilibration" ./system/ -o production.in

# Verbose output with all assumption details
lammps-reaper generate "NVT simulation at 500K" ./my_sim/ -v
```

#### Validate Existing Decks

```bash
# Validate all .in files in a directory (uses .data files as context)
lammps-reaper validate ./my_simulation/

# Validate a single file
lammps-reaper validate ./my_simulation/equil.in

# Specific validation level only
lammps-reaper validate ./my_simulation/ -l L1  # Syntax/physics only
lammps-reaper validate ./my_simulation/ -l L2  # Engine acceptance only
```

#### Analyze Files

```bash
# Analyze all LAMMPS files in a directory
lammps-reaper analyze ./my_simulation/

# Analyze a single file
lammps-reaper analyze ./my_simulation/system.data

# JSON output
lammps-reaper analyze ./my_simulation/ --json
```

### Python API

```python
import asyncio
from lammps_reaper import ReaperInput, generate_deck, discover_files
from pathlib import Path

# Discover files in a directory
discovered = discover_files(Path("./my_simulation"))
print(discovered.summary())

# Generate a deck
async def main():
    # Files are automatically discovered from the directory
    all_files = discovered.data_files + discovered.input_files

    reaper_input = ReaperInput(
        intent="Create LJ fluid NVT simulation at T=1.5, 500 atoms, 2000 steps",
        files=all_files,
        output_path=Path("./my_simulation/generated.in"),
        max_retries=3,
    )

    result = await generate_deck(reaper_input)

    print(f"Success: {result.success}")
    print(f"Attempts: {result.total_attempts}")

    # See what assumptions were made
    for assumption in result.assumptions:
        print(f"[{assumption.category.value}] {assumption.description}")
        print(f"  Value: {assumption.assumed_value}")

asyncio.run(main())
```

#### Direct File Discovery

```python
from lammps_reaper import discover_files, classify_file
from pathlib import Path

# Discover all LAMMPS files
discovered = discover_files(Path("./my_project"))

print(f"Data files: {[f.name for f in discovered.data_files]}")
print(f"Input scripts: {[f.name for f in discovered.input_files]}")
print(f"Potential files: {[f.name for f in discovered.potential_files]}")

# Get the primary data file
if discovered.primary_data_file:
    print(f"Primary data: {discovered.primary_data_file}")

# Get all context files (inputs + potentials)
context = discovered.context_files
```

#### Data File Analysis

```python
from lammps_reaper import analyze_data_file
from pathlib import Path

content = Path("system.data").read_text()
analysis = analyze_data_file(content)

print(f"Has bonds: {analysis['has_bonds']} ({analysis['bond_types']} types)")
print(f"Has angles: {analysis['has_angles']} ({analysis['angle_types']} types)")
print(f"Has charges: {analysis['has_charges']}")
print(f"Units hint: {analysis['units_hint']}")
```

## Validation Levels

| Level | Name | Description |
|-------|------|-------------|
| **L0** | Placeholder | Detects unresolved placeholders (`{{VAR}}`, `<VAR>`, `TODO:`, `FIXME:`) |
| **L1** | Syntax/Physics | Validates LAMMPS syntax, required commands, physics parameters |
| **L2** | Engine | Runs LAMMPS with `run 0` to verify engine accepts the script |
| **L3** | Execution | Runs ~20 timesteps to catch explosions and thermo issues |

## Project Structure

```
lammps_reaper/
├── __init__.py          # Package exports
├── cli.py               # Command-line interface
├── discovery.py         # File discovery module
├── generator.py         # LLM-based deck generation
├── provider.py          # Anthropic API wrapper
├── schemas.py           # Data structures (ReaperInput, ReaperOutput, etc.)
├── validation/          # Validation modules
│   ├── __init__.py      # Validation exports
│   ├── l0_placeholders.py
│   ├── l1_syntax.py
│   ├── l2_engine.py
│   ├── l3_physics.py
│   └── file_utils.py    # File handling for validation
├── tests/               # Test suite
├── pyproject.toml       # Package configuration
└── README.md            # This file
```

## Example: Multi-Phase Equilibration

Here's an example generating an equilibration workflow:

```bash
# Point at your project directory containing .data and .in files
lammps-reaper generate \
    "Create a complete equilibration workflow for this system.
     Include: 1. Energy minimization 2. NPT at 298.15K, 1 atm 3. NVT at 298.15K" \
    ./my_simulation/ -v
```

The LLM will:
1. Analyze the data file to detect topology (bonds, angles, dihedrals, impropers, charges)
2. Infer the required styles from the data file header
3. Generate a complete three-phase equilibration script
4. Validate the script actually runs in LAMMPS
5. Report all assumptions made (e.g., timestep, damping constants, run lengths)

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key (required) | None |
| `LAMMPS_BINARY` | Path to LAMMPS executable | Auto-detect |
| `ANTHROPIC_MODEL` | Claude model to use | `claude-sonnet-4-20250514` |

### ReaperInput Options

```python
ReaperInput(
    intent: str,                      # Natural language description (required)
    files: List[Path] = [],           # Data files and context files
    output_path: Optional[Path] = None,  # Where to write the deck
    lammps_binary: Optional[Path] = None,  # LAMMPS path for validation
    max_retries: int = 3,             # Retry attempts on failure
    enable_iterative_fixing: bool = True,  # Auto-fix on validation failure
)
```

## Testing

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=lammps_reaper

# Run live API tests (requires ANTHROPIC_API_KEY)
pytest tests/test_live_api.py -v

# Run specific test
pytest tests/test_discovery.py -v
```

## How It Works

1. **File Discovery**: Scans directory for LAMMPS files (data, inputs, potentials)
2. **Data Analysis**: Data files are analyzed to detect topology and required styles
3. **LLM Generation**: Claude generates a LAMMPS script with explicit assumptions
4. **Validation**: Script is validated through L0-L3 checks
5. **Iterative Fixing**: If validation fails, errors are fed back to the LLM for correction
6. **Output**: Final script and all metadata (assumptions, validation results) are returned

### The Prompt

The LLM receives:
- System prompt with LAMMPS best practices and output format requirements
- User's intent (natural language description)
- File contents (data files, example scripts)
- Data file analysis (detected topology, required styles)
- Instructions to output assumptions in JSON format

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [Claude](https://anthropic.com) by Anthropic
- Designed for [LAMMPS](https://lammps.org) molecular dynamics simulator
