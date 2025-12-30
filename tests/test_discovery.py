"""Tests for the file discovery module."""

import pytest
from pathlib import Path
import tempfile
import os

from lammps_reaper.discovery import (
    classify_file,
    discover_files,
    generate_output_filename,
    DiscoveredFiles,
    DATA_EXTENSIONS,
    INPUT_EXTENSIONS,
    POTENTIAL_EXTENSIONS,
)


class TestClassifyFile:
    """Tests for the classify_file function."""

    def test_data_extensions(self):
        """Test that data file extensions are classified correctly."""
        assert classify_file(Path("system.data")) == "data"
        assert classify_file(Path("polymer.dat")) == "data"
        assert classify_file(Path("test.DATA")) == "data"

    def test_input_extensions(self):
        """Test that input file extensions are classified correctly."""
        assert classify_file(Path("run.in")) == "input"
        assert classify_file(Path("equilibrate.lmp")) == "input"
        assert classify_file(Path("production.lammps")) == "input"
        assert classify_file(Path("minimize.inp")) == "input"

    def test_potential_extensions(self):
        """Test that potential file extensions are classified correctly."""
        assert classify_file(Path("Cu.eam")) == "potential"
        assert classify_file(Path("Si.tersoff")) == "potential"
        assert classify_file(Path("SiC.sw")) == "potential"
        assert classify_file(Path("params.meam")) == "potential"
        assert classify_file(Path("ffield.reax")) == "potential"
        assert classify_file(Path("pair.table")) == "potential"

    def test_compound_eam_extensions(self):
        """Test compound EAM extensions like .eam.fs and .eam.alloy."""
        assert classify_file(Path("Fe_Ni.eam.fs")) == "potential"
        assert classify_file(Path("Cu_Al.eam.alloy")) == "potential"
        assert classify_file(Path("Mishin.EAM.FS")) == "potential"

    def test_restart_extensions(self):
        """Test restart file extensions."""
        assert classify_file(Path("checkpoint.restart")) == "restart"
        assert classify_file(Path("state.rst")) == "restart"

    def test_other_extensions(self):
        """Test that unknown extensions are classified as 'other'."""
        assert classify_file(Path("README.md")) == "other"
        assert classify_file(Path("script.py")) == "other"
        assert classify_file(Path("config.json")) == "other"


class TestDiscoverFiles:
    """Tests for the discover_files function."""

    def test_discover_in_assets_directory(self):
        """Test discovering files in the assets directory."""
        assets_dir = Path(__file__).parent.parent / "assets"
        if not assets_dir.exists():
            pytest.skip("Assets directory not found")

        result = discover_files(assets_dir)

        assert isinstance(result, DiscoveredFiles)
        assert result.directory == assets_dir.resolve()
        # Check that we found the data file
        assert len(result.data_files) >= 1
        assert any(f.name == "equil_nvt_dry.data" for f in result.data_files)
        # Check that we found input scripts
        assert len(result.input_files) >= 3

    def test_discover_with_temp_directory(self):
        """Test discovering files in a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "system.data").write_text("# data file")
            (tmppath / "run.in").write_text("# input script")
            (tmppath / "cu.eam").write_text("# potential")
            (tmppath / "state.restart").write_text("# restart")
            (tmppath / "README.md").write_text("# readme")
            (tmppath / ".hidden").write_text("# hidden")

            result = discover_files(tmppath)

            assert len(result.data_files) == 1
            assert len(result.input_files) == 1
            assert len(result.potential_files) == 1
            assert len(result.restart_files) == 1
            # Hidden files and 'other' files should not be included by default
            assert len(result.other_files) == 0

    def test_discover_with_hidden_files(self):
        """Test discovering hidden files when include_hidden=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            (tmppath / "visible.in").write_text("# visible")
            (tmppath / ".hidden.in").write_text("# hidden")

            # Without hidden files
            result = discover_files(tmppath, include_hidden=False)
            assert len(result.input_files) == 1

            # With hidden files
            result = discover_files(tmppath, include_hidden=True)
            assert len(result.input_files) == 2

    def test_discover_empty_directory(self):
        """Test discovering files in an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_files(Path(tmpdir))

            assert len(result.data_files) == 0
            assert len(result.input_files) == 0
            assert len(result.potential_files) == 0
            assert len(result.all_files) == 0

    def test_discover_nonexistent_directory(self):
        """Test that discover_files raises error for nonexistent directory."""
        with pytest.raises(ValueError, match="does not exist"):
            discover_files(Path("/nonexistent/path"))

    def test_discover_file_instead_of_directory(self):
        """Test that discover_files raises error for file instead of directory."""
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError, match="not a directory"):
                discover_files(Path(f.name))


class TestDiscoveredFiles:
    """Tests for the DiscoveredFiles dataclass."""

    def test_primary_data_file(self):
        """Test that primary_data_file returns first data file."""
        result = DiscoveredFiles(
            directory=Path("/tmp"),
            data_files=[Path("a.data"), Path("b.data")],
        )
        assert result.primary_data_file == Path("a.data")

    def test_primary_data_file_empty(self):
        """Test that primary_data_file returns None when empty."""
        result = DiscoveredFiles(directory=Path("/tmp"))
        assert result.primary_data_file is None

    def test_context_files(self):
        """Test that context_files returns inputs + potentials."""
        result = DiscoveredFiles(
            directory=Path("/tmp"),
            input_files=[Path("run.in")],
            potential_files=[Path("cu.eam")],
        )
        assert len(result.context_files) == 2
        assert Path("run.in") in result.context_files
        assert Path("cu.eam") in result.context_files

    def test_all_files(self):
        """Test that all_files returns all discovered files."""
        result = DiscoveredFiles(
            directory=Path("/tmp"),
            data_files=[Path("a.data")],
            input_files=[Path("run.in")],
            potential_files=[Path("cu.eam")],
            restart_files=[Path("state.restart")],
        )
        assert len(result.all_files) == 4

    def test_summary(self):
        """Test that summary returns readable text."""
        result = DiscoveredFiles(
            directory=Path("/tmp/test"),
            data_files=[Path("system.data")],
            input_files=[Path("run.in")],
        )
        summary = result.summary()

        assert "Directory:" in summary
        assert "Data files (1)" in summary
        assert "system.data" in summary
        assert "Input scripts (1)" in summary
        assert "run.in" in summary

    def test_summary_empty(self):
        """Test that summary handles empty results."""
        result = DiscoveredFiles(directory=Path("/tmp/empty"))
        summary = result.summary()
        assert "No LAMMPS files found" in summary


class TestGenerateOutputFilename:
    """Tests for the generate_output_filename function."""

    def test_default_filename(self):
        """Test default filename generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_output_filename(Path(tmpdir))
            assert path.name == "generated.in"
            assert path.parent == Path(tmpdir)

    def test_custom_prefix(self):
        """Test custom prefix filename generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_output_filename(Path(tmpdir), prefix="equilibrate")
            assert path.name == "equilibrate.in"

    def test_increments_on_existing(self):
        """Test that filename increments if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create existing files
            (tmppath / "generated.in").write_text("")
            (tmppath / "generated_1.in").write_text("")

            path = generate_output_filename(tmppath)
            assert path.name == "generated_2.in"


class TestExtensionSets:
    """Tests to verify extension sets are correctly defined."""

    def test_data_extensions(self):
        """Test that data extensions include expected values."""
        assert ".data" in DATA_EXTENSIONS
        assert ".dat" in DATA_EXTENSIONS

    def test_input_extensions(self):
        """Test that input extensions include expected values."""
        assert ".in" in INPUT_EXTENSIONS
        assert ".lmp" in INPUT_EXTENSIONS
        assert ".lammps" in INPUT_EXTENSIONS
        assert ".inp" in INPUT_EXTENSIONS

    def test_potential_extensions(self):
        """Test that potential extensions include expected values."""
        assert ".eam" in POTENTIAL_EXTENSIONS
        assert ".tersoff" in POTENTIAL_EXTENSIONS
        assert ".sw" in POTENTIAL_EXTENSIONS
        assert ".meam" in POTENTIAL_EXTENSIONS
        assert ".reax" in POTENTIAL_EXTENSIONS
