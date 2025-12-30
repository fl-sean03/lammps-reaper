"""LAMMPS Reaper - LAMMPS deck generation and validation package.

This package provides tools for generating and validating LAMMPS input decks
using LLM-assisted generation with multi-level validation.
"""

from .schemas import (
    Assumption,
    AssumptionCategory,
    FileContext,
    GenerationAttempt,
    L0Result,
    L1Result,
    L2Result,
    L3Result,
    ThermoData,
    ReaperInput,
    ReaperOutput,
    ValidationResult,
)
from .generator import (
    LAMMPS_SYSTEM_PROMPT,
    analyze_data_file,
    build_file_context,
    build_prompt,
    clean_llm_output,
    detect_file_type,
    generate_deck,
    generate_deck_sync,
    parse_llm_response,
)
from .validation import (
    find_lammps_binary,
    validate_deck,
    validate_l0,
    validate_l1,
    validate_l2,
    validate_l3,
    # File utilities
    parse_file_references,
    find_file_in_context,
    setup_working_directory,
    cleanup_working_directory,
)
from .discovery import (
    DiscoveredFiles,
    discover_files,
    classify_file,
    generate_output_filename,
)


__version__ = "0.5.0"


__all__ = [
    # Schemas
    "Assumption",
    "AssumptionCategory",
    "FileContext",
    "GenerationAttempt",
    "L0Result",
    "L1Result",
    "L2Result",
    "L3Result",
    "ThermoData",
    "ReaperInput",
    "ReaperOutput",
    "ValidationResult",
    # Generator
    "LAMMPS_SYSTEM_PROMPT",
    "analyze_data_file",
    "build_file_context",
    "build_prompt",
    "clean_llm_output",
    "detect_file_type",
    "generate_deck",
    "generate_deck_sync",
    "parse_llm_response",
    # Validation
    "find_lammps_binary",
    "validate_deck",
    "validate_l0",
    "validate_l1",
    "validate_l2",
    "validate_l3",
    # File utilities
    "parse_file_references",
    "find_file_in_context",
    "setup_working_directory",
    "cleanup_working_directory",
    # Discovery
    "DiscoveredFiles",
    "discover_files",
    "classify_file",
    "generate_output_filename",
]
