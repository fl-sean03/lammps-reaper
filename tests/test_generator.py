"""Tests for lammps_reaper generator module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from lammps_reaper.generator import (
    detect_file_type,
    build_file_context,
    build_prompt,
    clean_llm_output,
    analyze_data_file,
    parse_llm_response,
)
from lammps_reaper.schemas import ReaperInput, AssumptionCategory


class TestDetectFileType:
    """Tests for detect_file_type function."""

    def test_detects_data_file_by_extension(self):
        """Test detection of LAMMPS data files by extension."""
        assert detect_file_type(Path("test.data"), "") == "data_file"
        assert detect_file_type(Path("test.dat"), "") == "data_file"

    def test_detects_input_file_by_extension(self):
        """Test detection of LAMMPS input files by extension."""
        assert detect_file_type(Path("test.in"), "") == "input_file"
        assert detect_file_type(Path("test.lmp"), "") == "input_file"
        assert detect_file_type(Path("test.lammps"), "") == "input_file"
        assert detect_file_type(Path("test.inp"), "") == "input_file"

    def test_detects_eam_potential_by_extension(self):
        """Test detection of EAM potential files by extension."""
        assert detect_file_type(Path("Cu.eam"), "") == "eam_potential"

    def test_detects_tersoff_potential_by_extension(self):
        """Test detection of Tersoff potential files by extension."""
        assert detect_file_type(Path("SiC.tersoff"), "") == "tersoff_potential"

    def test_detects_sw_potential_by_extension(self):
        """Test detection of Stillinger-Weber potential files by extension."""
        assert detect_file_type(Path("Si.sw"), "") == "sw_potential"

    def test_detects_meam_potential_by_extension(self):
        """Test detection of MEAM potential files by extension."""
        assert detect_file_type(Path("library.meam"), "") == "meam_potential"

    def test_detects_potential_by_name(self):
        """Test detection of potential files by naming convention."""
        assert detect_file_type(Path("potential_Fe.txt"), "") == "potential_file"
        assert detect_file_type(Path("Cu_pot.txt"), "") == "potential_file"

    def test_detects_parameter_file_by_name(self):
        """Test detection of parameter files by naming convention."""
        assert detect_file_type(Path("param_file.txt"), "") == "parameter_file"
        assert detect_file_type(Path("parameters.txt"), "") == "parameter_file"

    def test_detects_data_file_by_content(self):
        """Test detection of LAMMPS data file by content."""
        content = """LAMMPS data file
100 atoms
3 atom types
0.0 10.0 xlo xhi
Masses
"""
        assert detect_file_type(Path("file.txt"), content) == "data_file"

    def test_detects_input_file_by_content(self):
        """Test detection of LAMMPS input file by content."""
        content = """# LAMMPS input script
units lj
atom_style atomic
pair_style lj/cut 2.5
"""
        assert detect_file_type(Path("file.txt"), content) == "input_file"

    def test_detects_eam_by_content(self):
        """Test detection of EAM potential by content patterns."""
        content = """# EAM potential for Cu
nrho 500 drho 0.01
nr 500 dr 0.01
"""
        assert detect_file_type(Path("file.txt"), content) == "eam_potential"

    def test_returns_unknown_for_unrecognized(self):
        """Test returns 'unknown' for unrecognized files."""
        assert detect_file_type(Path("random.xyz"), "random content") == "unknown"


class TestAnalyzeDataFile:
    """Tests for analyze_data_file function."""

    def test_detects_bonds(self):
        """Test detection of bonds in data file."""
        content = """100 atoms
50 bonds
2 bond types
"""
        result = analyze_data_file(content)
        assert result["has_bonds"] is True
        assert result["bond_types"] == 2

    def test_detects_angles(self):
        """Test detection of angles in data file."""
        content = """100 atoms
30 angles
5 angle types
"""
        result = analyze_data_file(content)
        assert result["has_angles"] is True
        assert result["angle_types"] == 5

    def test_detects_dihedrals(self):
        """Test detection of dihedrals in data file."""
        content = """100 atoms
20 dihedrals
3 dihedral types
"""
        result = analyze_data_file(content)
        assert result["has_dihedrals"] is True
        assert result["dihedral_types"] == 3

    def test_detects_impropers(self):
        """Test detection of impropers in data file."""
        content = """100 atoms
10 impropers
2 improper types
"""
        result = analyze_data_file(content)
        assert result["has_impropers"] is True
        assert result["improper_types"] == 2

    def test_detects_units_hint(self):
        """Test detection of units hint in data file."""
        content = """LAMMPS data file, units = real
100 atoms
"""
        result = analyze_data_file(content)
        assert result["units_hint"] == "real"

    def test_detects_charges(self):
        """Test detection of charges in data file."""
        content = """100 atoms
Atoms # full
"""
        result = analyze_data_file(content)
        assert result["has_charges"] is True


class TestBuildFileContext:
    """Tests for build_file_context function."""

    def test_returns_empty_for_no_files(self):
        """Test returns empty tuple when no files provided."""
        context, analysis = build_file_context([])
        assert context == ""
        assert analysis == {}

    def test_includes_file_content(self, tmp_path):
        """Test includes file content in context."""
        test_file = tmp_path / "test.data"
        test_file.write_text("100 atoms\n3 atom types\n0.0 10.0 xlo xhi\n")

        context, analysis = build_file_context([test_file])
        assert "100 atoms" in context
        assert "test.data" in context

    def test_includes_file_type(self, tmp_path):
        """Test includes detected file type in context."""
        test_file = tmp_path / "test.data"
        test_file.write_text("100 atoms\n")

        context, analysis = build_file_context([test_file])
        assert "Type: data_file" in context

    def test_handles_nonexistent_files(self, tmp_path):
        """Test handles nonexistent files gracefully."""
        fake_path = tmp_path / "nonexistent.data"
        context, analysis = build_file_context([fake_path])
        assert context == ""
        assert analysis == {}

    def test_handles_multiple_files(self, tmp_path):
        """Test handles multiple files."""
        file1 = tmp_path / "data.data"
        file2 = tmp_path / "input.in"
        file1.write_text("100 atoms")
        file2.write_text("units lj")

        context, analysis = build_file_context([file1, file2])
        assert "data.data" in context
        assert "input.in" in context
        assert "100 atoms" in context
        assert "units lj" in context

    def test_truncates_large_files(self, tmp_path):
        """Test truncates files over MAX_FILE_LINES."""
        large_file = tmp_path / "large.data"
        # Create file with more than 1000 lines
        lines = [f"line {i}" for i in range(1500)]
        large_file.write_text("\n".join(lines))

        context, analysis = build_file_context([large_file])
        assert "truncated" in context.lower()
        assert "500 more lines" in context  # 1500 - 1000 = 500

    def test_includes_file_path(self, tmp_path):
        """Test includes full file path in context."""
        test_file = tmp_path / "test.data"
        test_file.write_text("content")

        context, analysis = build_file_context([test_file])
        assert f"Path: {test_file}" in context

    def test_returns_data_analysis(self, tmp_path):
        """Test returns data file analysis."""
        test_file = tmp_path / "test.data"
        test_file.write_text("100 atoms\n50 bonds\n2 bond types\n")

        context, analysis = build_file_context([test_file])
        assert analysis.get("has_bonds") is True
        assert analysis.get("bond_types") == 2


class TestBuildPrompt:
    """Tests for build_prompt function."""

    def test_includes_intent(self):
        """Test includes user intent in prompt."""
        prompt = build_prompt("Create NVT simulation for copper")
        assert "Create NVT simulation for copper" in prompt
        assert "SIMULATION REQUEST" in prompt

    def test_includes_file_context(self):
        """Test includes file context when provided."""
        file_context = "=== PROVIDED FILES ===\ntest content"
        prompt = build_prompt("Run simulation", file_context)
        assert "PROVIDED FILES" in prompt
        assert "test content" in prompt

    def test_includes_output_instructions(self):
        """Test includes output format instructions."""
        prompt = build_prompt("Test")
        assert "OUTPUT INSTRUCTIONS" in prompt
        assert "JSON" in prompt  # Now requests JSON assumptions

    def test_handles_empty_intent(self):
        """Test handles empty intent string."""
        prompt = build_prompt("")
        assert "SIMULATION REQUEST" in prompt

    def test_handles_none_file_context(self):
        """Test handles None file context."""
        prompt = build_prompt("Test", None)
        assert "PROVIDED FILES" not in prompt


class TestParseLlmResponse:
    """Tests for parse_llm_response function."""

    def test_parses_assumptions_json(self):
        """Test parses JSON assumptions block."""
        response = '''```json
{
  "assumptions": [
    {
      "category": "force_field",
      "description": "Assumed harmonic bonds",
      "assumed_value": "bond_style harmonic",
      "reasoning": "Common default",
      "confidence": "high"
    }
  ]
}
```

# LAMMPS INPUT SCRIPT
units lj
run 100
'''
        deck, assumptions = parse_llm_response(response)
        assert "units lj" in deck
        assert len(assumptions) == 1
        assert assumptions[0].category == AssumptionCategory.FORCE_FIELD
        assert assumptions[0].description == "Assumed harmonic bonds"

    def test_extracts_script_without_json(self):
        """Test extracts script when no JSON present."""
        response = '''# LAMMPS INPUT SCRIPT
units lj
atom_style atomic
run 100
'''
        deck, assumptions = parse_llm_response(response)
        assert "units lj" in deck
        assert "atom_style atomic" in deck
        assert len(assumptions) == 0

    def test_handles_malformed_json(self):
        """Test handles malformed JSON gracefully."""
        response = '''```json
{ invalid json }
```

# LAMMPS INPUT SCRIPT
units lj
'''
        deck, assumptions = parse_llm_response(response)
        assert "units lj" in deck
        assert len(assumptions) == 0


class TestCleanLlmOutput:
    """Tests for clean_llm_output function."""

    def test_removes_lammps_code_fence(self):
        """Test removes ```lammps code fences."""
        response = """```lammps
units lj
atom_style atomic
run 100
```"""
        result = clean_llm_output(response)
        assert "```" not in result
        assert "units lj" in result

    def test_removes_uppercase_lammps_fence(self):
        """Test removes ```LAMMPS code fences."""
        response = """```LAMMPS
units lj
```"""
        result = clean_llm_output(response)
        assert "```" not in result
        assert "units lj" in result

    def test_removes_lmp_code_fence(self):
        """Test removes ```lmp code fences."""
        response = """```lmp
units metal
```"""
        result = clean_llm_output(response)
        assert "```" not in result

    def test_removes_plain_code_fence(self):
        """Test removes plain ``` code fences."""
        response = """```
units lj
atom_style atomic
```"""
        result = clean_llm_output(response)
        assert "```" not in result
        assert "units lj" in result

    def test_preserves_content_without_fences(self):
        """Test preserves content when no fences present."""
        response = """units lj
atom_style atomic
pair_style lj/cut 2.5
run 100"""
        result = clean_llm_output(response)
        assert result == response

    def test_normalizes_line_endings(self):
        """Test normalizes Windows line endings."""
        response = "units lj\r\natom_style atomic\r\n"
        result = clean_llm_output(response)
        assert "\r\n" not in result
        assert "\r" not in result

    def test_strips_whitespace(self):
        """Test strips leading/trailing whitespace."""
        response = "\n\n  units lj\natom_style atomic  \n\n"
        result = clean_llm_output(response)
        assert result.startswith("units")
        assert result.endswith("atomic")

    def test_handles_empty_response(self):
        """Test handles empty response."""
        result = clean_llm_output("")
        assert result == ""

    def test_handles_only_fences(self):
        """Test handles response with only code fences."""
        response = "```lammps\n```"
        result = clean_llm_output(response)
        assert result == ""

    def test_preserves_internal_structure(self):
        """Test preserves internal whitespace and structure."""
        response = """units lj

# This is a comment

atom_style atomic
run 100"""
        result = clean_llm_output(response)
        assert "\n\n" in result  # Blank line preserved
        assert "# This is a comment" in result


class TestReaperInputIntegration:
    """Integration tests for ReaperInput with generator functions."""

    def test_build_context_from_reaper_input(self, tmp_path):
        """Test building context from ReaperInput files."""
        data_file = tmp_path / "data.data"
        data_file.write_text("100 atoms\n0.0 10.0 xlo xhi\n")

        ri = ReaperInput(
            intent="Run NVT simulation",
            files=[data_file],
        )

        context, analysis = build_file_context(ri.files)
        assert "data.data" in context
        assert "100 atoms" in context

    def test_build_prompt_from_reaper_input(self, tmp_path):
        """Test building prompt from ReaperInput."""
        data_file = tmp_path / "data.data"
        data_file.write_text("100 atoms")

        ri = ReaperInput(
            intent="Create equilibration simulation",
            files=[data_file],
        )

        context, _ = build_file_context(ri.files)
        prompt = build_prompt(ri.intent, context)

        assert "equilibration simulation" in prompt
        assert "100 atoms" in prompt
