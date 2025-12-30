"""Tests for lammps_reaper schemas."""

import pytest
from pathlib import Path

from lammps_reaper.schemas import (
    FileContext,
    L0Result,
    L1Result,
    L2Result,
    L3Result,
    ReaperInput,
    ReaperOutput,
    ValidationResult,
)


class TestFileContext:
    """Tests for FileContext dataclass."""

    def test_instantiation(self):
        """Test basic FileContext creation."""
        fc = FileContext(
            path=Path("/test/file.data"),
            content="test content",
            file_type="data_file",
        )
        assert fc.path == Path("/test/file.data")
        assert fc.content == "test content"
        assert fc.file_type == "data_file"

    def test_to_dict(self):
        """Test FileContext serialization."""
        fc = FileContext(
            path=Path("/test/file.data"),
            content="test content",
            file_type="data_file",
        )
        result = fc.to_dict()
        assert result == {
            "path": "/test/file.data",
            "content": "test content",
            "file_type": "data_file",
        }


class TestL0Result:
    """Tests for L0Result dataclass."""

    def test_instantiation_passed(self):
        """Test L0Result creation when validation passes."""
        result = L0Result(passed=True)
        assert result.passed is True
        assert result.placeholders_found == []
        assert result.unresolved_count == 0
        assert result.details == []

    def test_instantiation_failed(self):
        """Test L0Result creation when validation fails."""
        result = L0Result(
            passed=False,
            placeholders_found=["{{VAR1}}", "{{VAR2}}"],
            unresolved_count=2,
            details=["Line 1: Unresolved placeholder {{VAR1}}"],
        )
        assert result.passed is False
        assert len(result.placeholders_found) == 2
        assert result.unresolved_count == 2

    def test_to_dict(self):
        """Test L0Result serialization."""
        result = L0Result(
            passed=False,
            placeholders_found=["{{VAR}}"],
            unresolved_count=1,
            details=["Found placeholder"],
        )
        d = result.to_dict()
        assert d["passed"] is False
        assert d["placeholders_found"] == ["{{VAR}}"]
        assert d["unresolved_count"] == 1
        assert d["details"] == ["Found placeholder"]


class TestL1Result:
    """Tests for L1Result dataclass (syntax + physics validation)."""

    def test_instantiation_passed(self):
        """Test L1Result creation when validation passes."""
        result = L1Result(passed=True)
        assert result.passed is True
        assert result.syntax_errors == []
        assert result.physics_warnings == []
        assert result.line_numbers == []

    def test_instantiation_failed(self):
        """Test L1Result creation when validation fails."""
        result = L1Result(
            passed=False,
            syntax_errors=["Missing units command"],
            physics_warnings=["Timestep too large"],
            line_numbers=[1, 5],
            details=["Error: Missing units"],
        )
        assert result.passed is False
        assert len(result.syntax_errors) == 1
        assert len(result.physics_warnings) == 1

    def test_to_dict(self):
        """Test L1Result serialization."""
        result = L1Result(
            passed=True,
            syntax_errors=[],
            physics_warnings=["Warning 1"],
            line_numbers=[],
            details=["Validation passed"],
        )
        d = result.to_dict()
        assert d["passed"] is True
        assert d["syntax_errors"] == []
        assert d["physics_warnings"] == ["Warning 1"]
        assert d["line_numbers"] == []


class TestL2Result:
    """Tests for L2Result dataclass."""

    def test_instantiation_passed(self):
        """Test L2Result creation when validation passes."""
        result = L2Result(
            passed=True,
            engine_output="LAMMPS output",
            return_code=0,
            execution_time=1.5,
        )
        assert result.passed is True
        assert result.return_code == 0
        assert result.execution_time == 1.5

    def test_instantiation_skipped(self):
        """Test L2Result when validation is skipped."""
        result = L2Result(
            passed=True,
            engine_output="",
            return_code=-1,
            execution_time=0.0,
            details=["LAMMPS binary not found"],
        )
        assert result.passed is True
        assert result.return_code == -1

    def test_to_dict(self):
        """Test L2Result serialization."""
        result = L2Result(
            passed=False,
            engine_output="ERROR: Invalid command",
            return_code=1,
            execution_time=0.5,
            details=["LAMMPS Error"],
        )
        d = result.to_dict()
        assert d["passed"] is False
        assert d["return_code"] == 1
        assert "ERROR" in d["engine_output"]


class TestThermoData:
    """Tests for ThermoData dataclass."""

    def test_instantiation(self):
        """Test ThermoData creation."""
        from lammps_reaper.schemas import ThermoData
        data = ThermoData(step=10, temp=300.0, press=1.0, pe=-100.0, ke=50.0, etotal=-50.0)
        assert data.step == 10
        assert data.temp == 300.0
        assert data.pe == -100.0

    def test_to_dict(self):
        """Test ThermoData serialization."""
        from lammps_reaper.schemas import ThermoData
        data = ThermoData(step=0, temp=1.0)
        d = data.to_dict()
        assert d["step"] == 0
        assert d["temp"] == 1.0


class TestL3Result:
    """Tests for L3Result dataclass (minimal step execution + thermo sanity)."""

    def test_instantiation_passed(self):
        """Test L3Result creation when validation passes."""
        result = L3Result(passed=True)
        assert result.passed is True
        assert result.engine_output == ""
        assert result.return_code == 0
        assert result.steps_run == 0
        assert result.thermo_data == []
        assert result.thermo_warnings == []

    def test_instantiation_with_execution(self):
        """Test L3Result with execution details."""
        from lammps_reaper.schemas import ThermoData
        result = L3Result(
            passed=True,
            engine_output="LAMMPS output...",
            return_code=0,
            execution_time=1.5,
            steps_run=20,
            thermo_data=[ThermoData(step=0, temp=1.0), ThermoData(step=20, temp=1.1)],
            thermo_warnings=["Thermodynamic sanity checks passed"],
            details=["LAMMPS minimal execution passed (20 steps)"],
        )
        assert result.passed is True
        assert result.steps_run == 20
        assert result.execution_time == 1.5
        assert len(result.thermo_data) == 2
        assert result.thermo_data[0].temp == 1.0

    def test_to_dict(self):
        """Test L3Result serialization."""
        from lammps_reaper.schemas import ThermoData
        result = L3Result(
            passed=True,
            engine_output="Output",
            return_code=0,
            execution_time=2.0,
            steps_run=20,
            thermo_data=[ThermoData(step=0, temp=1.0)],
            thermo_warnings=["Test warning"],
            details=["Detail 1"],
        )
        d = result.to_dict()
        assert d["passed"] is True
        assert d["engine_output"] == "Output"
        assert d["steps_run"] == 20
        assert len(d["thermo_data"]) == 1
        assert d["thermo_data"][0]["temp"] == 1.0
        assert d["thermo_warnings"] == ["Test warning"]


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_instantiation_all_passed(self):
        """Test ValidationResult when all levels pass."""
        result = ValidationResult(
            overall_passed=True,
            l0=L0Result(passed=True),
            l1=L1Result(passed=True),
            l2=L2Result(passed=True),
            l3=L3Result(passed=True),
            issues=[],
        )
        assert result.overall_passed is True
        assert result.l0.passed is True
        assert result.l1.passed is True
        assert result.l2.passed is True
        assert result.l3.passed is True

    def test_instantiation_some_failed(self):
        """Test ValidationResult when some levels fail."""
        result = ValidationResult(
            overall_passed=False,
            l0=L0Result(passed=False, unresolved_count=1),
            l1=L1Result(passed=True),
            l2=L2Result(passed=True),
            l3=L3Result(passed=True),
            issues=["L0: 1 unresolved placeholder(s) found"],
        )
        assert result.overall_passed is False
        assert result.l0.passed is False
        assert len(result.issues) == 1

    def test_to_dict_serializes_all_levels(self):
        """Test ValidationResult serialization includes all levels."""
        result = ValidationResult(
            overall_passed=True,
            l0=L0Result(passed=True, details=["L0 ok"]),
            l1=L1Result(passed=True, details=["L1 ok"]),
            l2=L2Result(passed=True, details=["L2 ok"]),
            l3=L3Result(passed=True, details=["L3 ok"]),
            issues=[],
        )
        d = result.to_dict()
        assert d["overall_passed"] is True
        assert "l0" in d
        assert "l1" in d
        assert "l2" in d
        assert "l3" in d
        assert d["l0"]["details"] == ["L0 ok"]


class TestReaperInput:
    """Tests for ReaperInput dataclass."""

    def test_instantiation_minimal(self):
        """Test ReaperInput with minimal arguments."""
        ri = ReaperInput(intent="Create NVT simulation")
        assert ri.intent == "Create NVT simulation"
        assert ri.files == []
        assert ri.output_path is None
        assert ri.lammps_binary is None

    def test_instantiation_full(self):
        """Test ReaperInput with all arguments."""
        ri = ReaperInput(
            intent="Create NVT simulation",
            files=[Path("/test/data.data"), Path("/test/potential.eam")],
            output_path=Path("/output/simulation.in"),
            lammps_binary=Path("/usr/bin/lmp"),
        )
        assert ri.intent == "Create NVT simulation"
        assert len(ri.files) == 2
        assert ri.output_path == Path("/output/simulation.in")
        assert ri.lammps_binary == Path("/usr/bin/lmp")

    def test_to_dict(self):
        """Test ReaperInput serialization."""
        ri = ReaperInput(
            intent="Test",
            files=[Path("/test/file.data")],
            output_path=Path("/out/test.in"),
            lammps_binary=None,
        )
        d = ri.to_dict()
        assert d["intent"] == "Test"
        assert d["files"] == ["/test/file.data"]
        assert d["output_path"] == "/out/test.in"
        assert d["lammps_binary"] is None


class TestReaperOutput:
    """Tests for ReaperOutput dataclass."""

    def test_instantiation_success(self):
        """Test ReaperOutput for successful generation."""
        ro = ReaperOutput(
            success=True,
            deck_content="units lj\natom_style atomic\n",
            output_path=Path("/out/test.in"),
            validation=None,
            errors=[],
            warnings=[],
        )
        assert ro.success is True
        assert "units lj" in ro.deck_content
        assert ro.output_path == Path("/out/test.in")

    def test_instantiation_failure(self):
        """Test ReaperOutput for failed generation."""
        ro = ReaperOutput(
            success=False,
            deck_content="",
            output_path=None,
            validation=None,
            errors=["LLM API error: Connection failed"],
            warnings=[],
        )
        assert ro.success is False
        assert ro.deck_content == ""
        assert len(ro.errors) == 1

    def test_to_dict_with_validation(self):
        """Test ReaperOutput serialization with validation."""
        validation = ValidationResult(
            overall_passed=True,
            l0=L0Result(passed=True),
            l1=L1Result(passed=True),
            l2=L2Result(passed=True),
            l3=L3Result(passed=True),
        )
        ro = ReaperOutput(
            success=True,
            deck_content="units lj",
            output_path=Path("/out/test.in"),
            validation=validation,
        )
        d = ro.to_dict()
        assert d["success"] is True
        assert d["deck_content"] == "units lj"
        assert d["output_path"] == "/out/test.in"
        assert d["validation"] is not None
        assert d["validation"]["overall_passed"] is True

    def test_to_dict_without_validation(self):
        """Test ReaperOutput serialization without validation."""
        ro = ReaperOutput(
            success=True,
            deck_content="units lj",
        )
        d = ro.to_dict()
        assert d["validation"] is None
        assert d["output_path"] is None
