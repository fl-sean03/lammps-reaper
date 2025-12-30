"""L0 Validation: Placeholder checks.

This module validates that all template placeholders have been resolved
in the generated LAMMPS deck.
"""

import re
from typing import List, Tuple

from ..schemas import L0Result

# Placeholder patterns to detect
PLACEHOLDER_PATTERNS = [
    (r"\{\{([A-Z_][A-Z0-9_]*)\}\}", "double_brace"),  # {{PLACEHOLDER}}
    (r"<([A-Z_][A-Z0-9_]*)>", "angle_bracket"),  # <PLACEHOLDER>
    (r"\bTODO:\s*(.+?)(?:\n|$)", "todo"),  # TODO: description
    (r"\bFIXME:\s*(.+?)(?:\n|$)", "fixme"),  # FIXME: description
    (r"\bXXX:\s*(.+?)(?:\n|$)", "xxx"),  # XXX: description
]


def _find_placeholders(content: str) -> List[Tuple[str, str, int]]:
    """Find all placeholders in the content.

    Args:
        content: The content to search for placeholders.

    Returns:
        List of tuples (placeholder_text, pattern_type, line_number).
    """
    placeholders = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, start=1):
        # Skip comment lines for some patterns (but still check for TODO/FIXME)
        is_comment = line.strip().startswith("#")

        for pattern, pattern_type in PLACEHOLDER_PATTERNS:
            # For TODO/FIXME/XXX, we want to find them even in comments
            # For template placeholders, we should find them everywhere
            if pattern_type in ("todo", "fixme", "xxx"):
                # These are typically in comments
                matches = re.finditer(pattern, line, re.IGNORECASE)
            else:
                # Template placeholders - find everywhere
                matches = re.finditer(pattern, line)

            for match in matches:
                full_match = match.group(0)
                placeholders.append((full_match, pattern_type, line_num))

    return placeholders


def validate_l0(content: str) -> L0Result:
    """Validate that no unresolved placeholders remain in the deck.

    Detects patterns like:
    - {{PLACEHOLDER}} - Double brace template variables
    - <PLACEHOLDER> - Angle bracket placeholders
    - TODO: - Incomplete work markers
    - FIXME: - Bug or issue markers
    - XXX: - Warning markers

    Args:
        content: The LAMMPS deck content to validate.

    Returns:
        L0Result with validation status and details.
    """
    placeholders = _find_placeholders(content)

    # Separate template placeholders from markers
    template_placeholders = [
        p for p in placeholders if p[1] in ("double_brace", "angle_bracket")
    ]
    markers = [p for p in placeholders if p[1] in ("todo", "fixme", "xxx")]

    # Build details list
    details = []
    placeholders_found = []

    for placeholder_text, pattern_type, line_num in template_placeholders:
        placeholders_found.append(placeholder_text)
        details.append(f"Line {line_num}: Unresolved placeholder {placeholder_text}")

    for marker_text, pattern_type, line_num in markers:
        placeholders_found.append(marker_text)
        details.append(f"Line {line_num}: {pattern_type.upper()} marker found: {marker_text}")

    # Validation passes only if no template placeholders found
    # Markers are warnings but don't fail validation
    passed = len(template_placeholders) == 0

    return L0Result(
        passed=passed,
        placeholders_found=placeholders_found,
        unresolved_count=len(template_placeholders),
        details=details,
    )
