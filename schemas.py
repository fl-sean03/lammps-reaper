"""Schemas for the lammps_reaper package.

This module defines all dataclasses used for input/output and validation results.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AssumptionCategory(Enum):
    """Categories of assumptions the LLM might make."""

    FORCE_FIELD = "force_field"  # pair_style, bond_style, etc.
    UNITS = "units"  # Unit system choice
    PARAMETERS = "parameters"  # Timestep, cutoffs, etc.
    TOPOLOGY = "topology"  # Bond/angle/dihedral types
    SIMULATION = "simulation"  # Run length, ensemble, etc.
    OUTPUT = "output"  # Dump, thermo settings
    OTHER = "other"


@dataclass
class Assumption:
    """Represents an assumption made by the LLM during generation.

    When the LLM doesn't have complete information, it must make assumptions.
    This class tracks what was assumed and why.
    """

    category: AssumptionCategory
    description: str
    assumed_value: str
    reasoning: str = ""
    confidence: str = "medium"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "description": self.description,
            "assumed_value": self.assumed_value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


@dataclass
class GenerationAttempt:
    """Record of a single generation attempt in iterative fixing."""

    attempt_number: int
    deck_content: str
    validation_passed: bool
    errors: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "attempt_number": self.attempt_number,
            "deck_content": self.deck_content,
            "validation_passed": self.validation_passed,
            "errors": self.errors,
            "fixes_applied": self.fixes_applied,
        }


@dataclass
class FileContext:
    """Represents a file with its content and type information."""

    path: Path
    content: str
    file_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "content": self.content,
            "file_type": self.file_type,
        }


@dataclass
class L0Result:
    """Level 0 validation result: Placeholder checks."""

    passed: bool
    placeholders_found: List[str] = field(default_factory=list)
    unresolved_count: int = 0
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "placeholders_found": self.placeholders_found,
            "unresolved_count": self.unresolved_count,
            "details": self.details,
        }


@dataclass
class L1Result:
    """Level 1 validation result: Syntax + Physics parameter checks."""

    passed: bool
    syntax_errors: List[str] = field(default_factory=list)
    physics_warnings: List[str] = field(default_factory=list)
    line_numbers: List[int] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "syntax_errors": self.syntax_errors,
            "physics_warnings": self.physics_warnings,
            "line_numbers": self.line_numbers,
            "details": self.details,
        }


@dataclass
class L2Result:
    """Level 2 validation result: Engine/LAMMPS execution checks."""

    passed: bool
    engine_output: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "engine_output": self.engine_output,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "details": self.details,
        }


@dataclass
class ThermoData:
    """Thermodynamic data parsed from LAMMPS output."""

    step: int = 0
    temp: Optional[float] = None
    press: Optional[float] = None
    pe: Optional[float] = None  # Potential energy
    ke: Optional[float] = None  # Kinetic energy
    etotal: Optional[float] = None  # Total energy

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step,
            "temp": self.temp,
            "press": self.press,
            "pe": self.pe,
            "ke": self.ke,
            "etotal": self.etotal,
        }


@dataclass
class L3Result:
    """Level 3 validation result: Minimal step execution + thermo sanity."""

    passed: bool
    engine_output: str = ""
    return_code: int = 0
    execution_time: float = 0.0
    steps_run: int = 0
    thermo_data: List[ThermoData] = field(default_factory=list)
    thermo_warnings: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "engine_output": self.engine_output,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "steps_run": self.steps_run,
            "thermo_data": [t.to_dict() for t in self.thermo_data],
            "thermo_warnings": self.thermo_warnings,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Complete validation result across all levels."""

    overall_passed: bool
    l0: L0Result
    l1: L1Result
    l2: L2Result
    l3: L3Result
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_passed": self.overall_passed,
            "l0": self.l0.to_dict(),
            "l1": self.l1.to_dict(),
            "l2": self.l2.to_dict(),
            "l3": self.l3.to_dict(),
            "issues": self.issues,
        }


@dataclass
class ReaperInput:
    """Input configuration for the LAMMPS deck generator."""

    intent: str
    files: List[Path] = field(default_factory=list)
    output_path: Optional[Path] = None
    lammps_binary: Optional[Path] = None
    max_retries: int = 3  # Maximum attempts for iterative fixing
    enable_iterative_fixing: bool = True  # Enable error feedback loop

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intent": self.intent,
            "files": [str(f) for f in self.files],
            "output_path": str(self.output_path) if self.output_path else None,
            "lammps_binary": str(self.lammps_binary) if self.lammps_binary else None,
            "max_retries": self.max_retries,
            "enable_iterative_fixing": self.enable_iterative_fixing,
        }


@dataclass
class ReaperOutput:
    """Output from the LAMMPS deck generator."""

    success: bool
    deck_content: str
    output_path: Optional[Path] = None
    validation: Optional[ValidationResult] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    assumptions: List[Assumption] = field(default_factory=list)
    attempts: List[GenerationAttempt] = field(default_factory=list)
    total_attempts: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "deck_content": self.deck_content,
            "output_path": str(self.output_path) if self.output_path else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "attempts": [a.to_dict() for a in self.attempts],
            "total_attempts": self.total_attempts,
        }

    def assumptions_summary(self) -> str:
        """Generate a human-readable summary of assumptions."""
        if not self.assumptions:
            return "No assumptions recorded."

        lines = ["Assumptions made during generation:"]
        for a in self.assumptions:
            lines.append(f"  [{a.category.value}] {a.description}")
            lines.append(f"    Value: {a.assumed_value}")
            if a.reasoning:
                lines.append(f"    Reason: {a.reasoning}")
            lines.append(f"    Confidence: {a.confidence}")
        return "\n".join(lines)
