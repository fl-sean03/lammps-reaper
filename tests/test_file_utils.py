"""Tests for lammps_reaper validation file utilities."""

import pytest
from pathlib import Path

from lammps_reaper.validation.file_utils import (
    parse_file_references,
    find_file_in_context,
    setup_working_directory,
    cleanup_working_directory,
)


class TestParseFileReferences:
    """Tests for parse_file_references function."""

    def test_extracts_read_data(self):
        """Test extracting read_data file references."""
        content = """units real
atom_style full
read_data system.data
"""
        refs = parse_file_references(content)
        assert "system.data" in refs

    def test_extracts_read_restart(self):
        """Test extracting read_restart file references."""
        content = """units metal
read_restart checkpoint.restart
"""
        refs = parse_file_references(content)
        assert "checkpoint.restart" in refs

    def test_extracts_include(self):
        """Test extracting include file references."""
        content = """units lj
include settings.lammps
pair_style lj/cut 2.5
"""
        refs = parse_file_references(content)
        assert "settings.lammps" in refs

    def test_extracts_molecule(self):
        """Test extracting molecule file references."""
        content = """units real
molecule water water.mol
"""
        refs = parse_file_references(content)
        assert "water.mol" in refs

    def test_extracts_pair_coeff_with_file(self):
        """Test extracting potential file from pair_coeff."""
        content = """units metal
pair_style eam
pair_coeff * * Cu_u3.eam
"""
        refs = parse_file_references(content)
        assert "Cu_u3.eam" in refs

    def test_extracts_tersoff_potential(self):
        """Test extracting Tersoff potential file."""
        content = """units metal
pair_style tersoff
pair_coeff * * SiC.tersoff Si C
"""
        refs = parse_file_references(content)
        assert "SiC.tersoff" in refs

    def test_ignores_comments(self):
        """Test that comment lines are ignored."""
        content = """# read_data commented.data
units lj
read_data actual.data  # inline comment
"""
        refs = parse_file_references(content)
        assert "commented.data" not in refs
        assert "actual.data" in refs

    def test_ignores_empty_lines(self):
        """Test that empty lines don't cause issues."""
        content = """units lj

read_data file.data

run 100
"""
        refs = parse_file_references(content)
        assert "file.data" in refs

    def test_multiple_references(self):
        """Test extracting multiple file references."""
        content = """units real
read_data system.data
include forces.lammps
pair_coeff * * potential.eam
"""
        refs = parse_file_references(content)
        assert "system.data" in refs
        assert "forces.lammps" in refs
        assert "potential.eam" in refs

    def test_no_references(self):
        """Test handling content with no file references."""
        content = """units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
"""
        refs = parse_file_references(content)
        assert len(refs) == 0


class TestFindFileInContext:
    """Tests for find_file_in_context function."""

    def test_finds_exact_match(self, tmp_path):
        """Test finding file by exact name match."""
        data_file = tmp_path / "data.data"
        data_file.write_text("test content")

        result = find_file_in_context("data.data", [data_file])
        assert result == data_file

    def test_finds_in_search_dirs(self, tmp_path):
        """Test finding file in additional search directories."""
        data_file = tmp_path / "data.data"
        data_file.write_text("test content")

        result = find_file_in_context(
            "data.data",
            [],
            search_dirs=[tmp_path]
        )
        assert result == data_file

    def test_returns_none_for_missing(self, tmp_path):
        """Test returns None when file not found."""
        result = find_file_in_context("nonexistent.data", [])
        assert result is None

    def test_matches_filename_from_path(self, tmp_path):
        """Test matching just the filename from a path reference."""
        data_file = tmp_path / "system.data"
        data_file.write_text("content")

        result = find_file_in_context(
            "subdir/system.data",
            [data_file]
        )
        assert result == data_file


class TestSetupWorkingDirectory:
    """Tests for setup_working_directory function."""

    def test_creates_deck_file(self, tmp_path):
        """Test that deck file is created in working directory."""
        work_dir = tmp_path / "work"
        deck_content = "units lj\nrun 100\n"

        deck_path = setup_working_directory(
            deck_content,
            [],
            work_dir,
        )

        assert deck_path.exists()
        assert deck_path.read_text() == deck_content
        assert deck_path.parent == work_dir

    def test_copies_referenced_data_file(self, tmp_path):
        """Test that referenced data file is copied."""
        # Create source data file
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        data_file = source_dir / "system.data"
        data_file.write_text("100 atoms\n")

        # Create deck referencing the file
        work_dir = tmp_path / "work"
        deck_content = "units real\nread_data system.data\n"

        setup_working_directory(
            deck_content,
            [data_file],
            work_dir,
        )

        # Check data file was copied
        copied_data = work_dir / "system.data"
        assert copied_data.exists()
        assert copied_data.read_text() == "100 atoms\n"

    def test_copies_multiple_files(self, tmp_path):
        """Test copying multiple referenced files."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        data_file = source_dir / "system.data"
        data_file.write_text("data content")

        potential_file = source_dir / "Cu.eam"
        potential_file.write_text("potential content")

        work_dir = tmp_path / "work"
        deck_content = """units metal
read_data system.data
pair_style eam
pair_coeff * * Cu.eam
"""

        setup_working_directory(
            deck_content,
            [data_file, potential_file],
            work_dir,
        )

        assert (work_dir / "system.data").exists()
        assert (work_dir / "Cu.eam").exists()

    def test_copies_unreferenced_context_files(self, tmp_path):
        """Test that context files are copied even if not explicitly referenced."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        # This file isn't referenced in the deck but should still be copied
        extra_file = source_dir / "extra.txt"
        extra_file.write_text("extra content")

        work_dir = tmp_path / "work"
        deck_content = "units lj\nrun 100\n"

        setup_working_directory(
            deck_content,
            [extra_file],
            work_dir,
        )

        assert (work_dir / "extra.txt").exists()


class TestCleanupWorkingDirectory:
    """Tests for cleanup_working_directory function."""

    def test_removes_directory(self, tmp_path):
        """Test complete directory removal."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "test.txt").write_text("content")

        cleanup_working_directory(work_dir)

        assert not work_dir.exists()

    def test_handles_nonexistent_directory(self, tmp_path):
        """Test handling nonexistent directory gracefully."""
        work_dir = tmp_path / "nonexistent"

        # Should not raise
        cleanup_working_directory(work_dir)

    def test_keep_outputs_removes_inputs(self, tmp_path):
        """Test keep_outputs mode keeps output files."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "input.lammps").write_text("deck")
        (work_dir / "log.lammps").write_text("output")

        cleanup_working_directory(work_dir, keep_outputs=True)

        # Input should be removed, directory and other files should remain
        assert work_dir.exists()
        assert not (work_dir / "input.lammps").exists()
