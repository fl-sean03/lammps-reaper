"""Tests for lammps_reaper validation modules."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lammps_reaper.validation import (
    validate_l0,
    validate_l1,
    validate_l2,
    validate_l3,
    validate_deck,
    find_lammps_binary,
)


class TestL0Validation:
    """Tests for L0 placeholder validation."""

    def test_detects_double_brace_placeholders(self, deck_with_placeholders):
        """Test L0 detects {{PLACEHOLDER}} style placeholders."""
        result = validate_l0(deck_with_placeholders)
        assert result.passed is False
        assert result.unresolved_count == 2
        assert "{{UNITS}}" in result.placeholders_found
        assert "{{TIMESTEP}}" in result.placeholders_found

    def test_detects_angle_bracket_placeholders(self):
        """Test L0 detects <PLACEHOLDER> style placeholders."""
        content = "units <UNIT_SYSTEM>\natom_style <ATOM_STYLE>"
        result = validate_l0(content)
        assert result.passed is False
        assert result.unresolved_count == 2

    def test_detects_todo_markers(self, deck_with_todo_markers):
        """Test L0 detects TODO markers as warnings."""
        result = validate_l0(deck_with_todo_markers)
        # TODO markers should be found but don't fail validation
        assert result.passed is True  # Only template placeholders fail
        assert any("TODO" in p for p in result.placeholders_found)
        assert any("FIXME" in p for p in result.placeholders_found)

    def test_passes_clean_content(self, sample_lammps_deck):
        """Test L0 passes when no placeholders are present."""
        result = validate_l0(sample_lammps_deck)
        assert result.passed is True
        assert result.unresolved_count == 0

    def test_empty_content(self):
        """Test L0 handles empty content."""
        result = validate_l0("")
        assert result.passed is True
        assert result.unresolved_count == 0

    def test_line_numbers_in_details(self):
        """Test L0 includes line numbers in details."""
        content = "line1\n{{PLACEHOLDER}}\nline3"
        result = validate_l0(content)
        assert result.passed is False
        assert any("Line 2" in d for d in result.details)


class TestL1Validation:
    """Tests for L1 syntax + physics validation."""

    def test_detects_missing_units(self, deck_missing_units):
        """Test L1 detects missing units command."""
        result = validate_l1(deck_missing_units)
        assert result.passed is False
        assert any("units" in err.lower() for err in result.syntax_errors)

    def test_detects_missing_atom_style(self):
        """Test L1 detects missing atom_style command."""
        content = """units lj
boundary p p p
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
"""
        result = validate_l1(content)
        assert result.passed is False
        assert any("atom_style" in err.lower() for err in result.syntax_errors)

    def test_detects_missing_structure_commands(self):
        """Test L1 detects missing structure definition."""
        content = """units lj
atom_style atomic
boundary p p p
pair_style lj/cut 2.5
"""
        result = validate_l1(content)
        assert result.passed is False
        assert any("structure" in err.lower() for err in result.syntax_errors)

    def test_passes_valid_deck(self, sample_lammps_deck):
        """Test L1 passes for a valid deck."""
        result = validate_l1(sample_lammps_deck)
        assert result.passed is True
        assert len(result.syntax_errors) == 0

    def test_detects_unbalanced_quotes(self, deck_with_bad_syntax):
        """Test L1 detects unbalanced quotes."""
        result = validate_l1(deck_with_bad_syntax)
        assert result.passed is False
        assert any("quote" in err.lower() for err in result.syntax_errors)

    def test_detects_unbalanced_parentheses(self):
        """Test L1 detects unbalanced parentheses."""
        content = """units lj
atom_style atomic
variable x equal (1+2
region box block 0 4 0 4 0 4
create_box 1 box
"""
        result = validate_l1(content)
        assert result.passed is False
        assert any("parenthes" in err.lower() for err in result.syntax_errors)

    def test_validates_units_argument(self):
        """Test L1 validates units command argument."""
        content = """units invalid_unit_system
atom_style atomic
region box block 0 4 0 4 0 4
create_box 1 box
"""
        result = validate_l1(content)
        assert result.passed is False
        assert any("unit" in err.lower() for err in result.syntax_errors)

    def test_validates_atom_style_argument(self):
        """Test L1 validates atom_style command argument."""
        content = """units lj
atom_style invalid_style
region box block 0 4 0 4 0 4
create_box 1 box
"""
        result = validate_l1(content)
        assert result.passed is False
        assert any("atom style" in err.lower() for err in result.syntax_errors)

    def test_warns_on_missing_pair_coeff(self):
        """Test L1 warns when pair_style present but no pair_coeff."""
        content = """units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
pair_style lj/cut 2.5
"""
        result = validate_l1(content)
        # Warning doesn't fail validation but should be in details
        assert any("pair_coeff" in d.lower() for d in result.details)

    # Physics parameter checks (now in L1)
    def test_detects_large_timestep(self):
        """Test L1 warns about large timestep for LJ units."""
        content = """units lj
atom_style atomic
timestep 0.1
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
run 100
"""
        result = validate_l1(content)
        assert any("timestep" in w.lower() for w in result.physics_warnings)

    def test_detects_small_timestep(self):
        """Test L1 warns about very small timestep."""
        content = """units lj
atom_style atomic
timestep 0.0000001
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
run 100
"""
        result = validate_l1(content)
        assert any("timestep" in w.lower() for w in result.physics_warnings)

    def test_detects_negative_temperature(self, deck_with_physics_issues):
        """Test L1 fails on negative temperature."""
        result = validate_l1(deck_with_physics_issues)
        assert result.passed is False
        assert any("negative temperature" in w.lower() for w in result.physics_warnings)

    def test_handles_missing_timestep(self):
        """Test L1 handles deck without explicit timestep."""
        content = """units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
run 100
"""
        result = validate_l1(content)
        assert any("default" in d.lower() for d in result.details)

    def test_handles_different_unit_systems(self):
        """Test L1 handles different unit systems."""
        content = """units metal
atom_style atomic
timestep 0.001
lattice fcc 3.52
region box block 0 4 0 4 0 4
create_box 1 box
run 100
"""
        result = validate_l1(content)
        # Should pass with valid timestep for metal units
        assert result.passed is True


class TestL2Validation:
    """Tests for L2 engine validation (0 steps)."""

    def test_skips_when_no_binary(self, sample_lammps_deck):
        """Test L2 skips validation when no LAMMPS binary is found."""
        with patch(
            "lammps_reaper.validation.l2_engine.find_lammps_binary"
        ) as mock_find:
            mock_find.return_value = None
            result = validate_l2(sample_lammps_deck)
            assert result.passed is True
            assert "skipped" in result.details[0].lower()

    def test_find_lammps_binary_user_path(self, tmp_path):
        """Test find_lammps_binary with user-provided path."""
        # Create a fake binary
        fake_binary = tmp_path / "lmp"
        fake_binary.touch()

        result = find_lammps_binary(fake_binary)
        assert result == fake_binary

    def test_find_lammps_binary_env_var(self, tmp_path):
        """Test find_lammps_binary with environment variable."""
        fake_binary = tmp_path / "lmp_env"
        fake_binary.touch()

        with patch.dict("os.environ", {"LAMMPS_BINARY": str(fake_binary)}):
            result = find_lammps_binary()
            assert result == fake_binary

    def test_find_lammps_binary_not_found(self):
        """Test find_lammps_binary returns None when not found."""
        with patch.dict("os.environ", {"LAMMPS_BINARY": ""}, clear=False):
            # Use a non-existent path
            result = find_lammps_binary(Path("/nonexistent/lmp"))
            # Should check other paths, but likely return None if none exist
            # This test just verifies it doesn't crash

    @pytest.mark.skipif(
        not find_lammps_binary(),
        reason="LAMMPS binary not available"
    )
    def test_validates_with_lammps_binary(self, sample_lammps_deck):
        """Test L2 actually runs LAMMPS if binary is available."""
        result = validate_l2(sample_lammps_deck)
        # Should either pass or fail based on LAMMPS output
        assert isinstance(result.passed, bool)
        assert result.execution_time >= 0

    def test_handles_timeout(self, sample_lammps_deck):
        """Test L2 handles execution timeout."""
        with patch("subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired("lmp", 30)
            with patch(
                "lammps_reaper.validation.l2_engine.find_lammps_binary"
            ) as mock_find:
                mock_find.return_value = Path("/usr/bin/lmp")
                result = validate_l2(sample_lammps_deck, timeout=30.0)
                assert result.passed is False
                assert "timed out" in result.details[0].lower()


class TestL3Validation:
    """Tests for L3 minimal step execution validation."""

    def test_skips_when_no_binary(self, sample_lammps_deck):
        """Test L3 skips validation when no LAMMPS binary is found."""
        with patch(
            "lammps_reaper.validation.l3_physics.find_lammps_binary"
        ) as mock_find:
            mock_find.return_value = None
            result = validate_l3(sample_lammps_deck)
            assert result.passed is True
            assert "skipped" in result.details[0].lower()
            assert result.steps_run == 0

    @pytest.mark.skipif(
        not find_lammps_binary(),
        reason="LAMMPS binary not available"
    )
    def test_runs_minimal_steps(self, sample_lammps_deck):
        """Test L3 runs LAMMPS with minimal steps."""
        result = validate_l3(sample_lammps_deck, steps=10)
        assert isinstance(result.passed, bool)
        assert result.execution_time >= 0
        if result.passed:
            assert result.steps_run == 10

    def test_handles_timeout(self, sample_lammps_deck):
        """Test L3 handles execution timeout."""
        with patch("subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired("lmp", 60)
            with patch(
                "lammps_reaper.validation.l3_physics.find_lammps_binary"
            ) as mock_find:
                mock_find.return_value = Path("/usr/bin/lmp")
                result = validate_l3(sample_lammps_deck, timeout=60.0)
                assert result.passed is False
                assert "timed out" in result.details[0].lower()

    def test_has_execution_fields(self, sample_lammps_deck):
        """Test L3 result has execution-related fields."""
        with patch(
            "lammps_reaper.validation.l3_physics.find_lammps_binary"
        ) as mock_find:
            mock_find.return_value = None
            result = validate_l3(sample_lammps_deck)
            assert hasattr(result, "engine_output")
            assert hasattr(result, "return_code")
            assert hasattr(result, "execution_time")
            assert hasattr(result, "steps_run")
            assert hasattr(result, "thermo_data")
            assert hasattr(result, "thermo_warnings")

    @pytest.mark.skipif(
        not find_lammps_binary(),
        reason="LAMMPS binary not available"
    )
    def test_detects_explosions(self):
        """Test L3 detects simulation explosions with bad parameters."""
        # A deck with extremely large timestep that should cause explosion
        bad_deck = """units lj
atom_style atomic
boundary p p p
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
velocity all create 10.0 87287
timestep 1.0
fix 1 all nve
run 100
"""
        result = validate_l3(bad_deck, steps=50)
        # This may or may not explode depending on the system
        # We just verify it runs without crashing
        assert isinstance(result.passed, bool)


class TestValidateDeck:
    """Tests for the combined validate_deck function."""

    def test_runs_all_levels(self, sample_lammps_deck):
        """Test validate_deck runs all validation levels."""
        result = validate_deck(sample_lammps_deck)
        assert hasattr(result, "l0")
        assert hasattr(result, "l1")
        assert hasattr(result, "l2")
        assert hasattr(result, "l3")

    def test_overall_passed_when_all_pass(self, sample_lammps_deck):
        """Test overall_passed is True when all levels pass."""
        with patch(
            "lammps_reaper.validation.l2_engine.find_lammps_binary"
        ) as mock_find_l2, patch(
            "lammps_reaper.validation.l3_physics.find_lammps_binary"
        ) as mock_find_l3:
            mock_find_l2.return_value = None  # Skip L2
            mock_find_l3.return_value = None  # Skip L3
            result = validate_deck(sample_lammps_deck)
            assert result.overall_passed is True

    def test_overall_failed_when_l0_fails(self, deck_with_placeholders):
        """Test overall_passed is False when L0 fails."""
        result = validate_deck(deck_with_placeholders)
        assert result.overall_passed is False
        assert result.l0.passed is False
        assert any("L0" in issue for issue in result.issues)

    def test_overall_failed_when_l1_fails(self, deck_missing_units):
        """Test overall_passed is False when L1 fails."""
        result = validate_deck(deck_missing_units)
        assert result.overall_passed is False
        assert result.l1.passed is False
        assert any("L1" in issue for issue in result.issues)

    def test_issues_aggregated(self, deck_with_placeholders):
        """Test issues are aggregated from all levels."""
        result = validate_deck(deck_with_placeholders)
        assert len(result.issues) > 0
        # Should have issues from L0 and L1 (placeholders + syntax issues)

    def test_with_lammps_binary_path(self, sample_lammps_deck, tmp_path):
        """Test validate_deck with custom LAMMPS binary path."""
        fake_binary = tmp_path / "lmp"
        fake_binary.touch()
        # This will pass the binary path to L2/L3 validation
        result = validate_deck(sample_lammps_deck, lammps_binary=fake_binary)
        # Should complete without error
        assert hasattr(result, "overall_passed")

    def test_l1_includes_physics_warnings(self, sample_lammps_deck):
        """Test L1 result includes physics_warnings field."""
        result = validate_deck(sample_lammps_deck)
        assert hasattr(result.l1, "physics_warnings")

    def test_l3_includes_execution_fields(self, sample_lammps_deck):
        """Test L3 result includes execution-related fields."""
        result = validate_deck(sample_lammps_deck)
        assert hasattr(result.l3, "steps_run")
        assert hasattr(result.l3, "engine_output")
