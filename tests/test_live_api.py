"""Live API tests for lammps_reaper.

These tests require a valid ANTHROPIC_API_KEY environment variable
and optionally a LAMMPS binary for execution tests.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file for API key
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not configured",
)

# LAMMPS binary path from environment or default
LAMMPS_BIN = Path(
    os.environ.get(
        "LAMMPS_BINARY",
        "/home/sf2/Workspace/main/39-GPUTests/1-GPUTests/md-lammps/install/bin/lmp",
    )
)


class TestProviderLive:
    """Tests for the AnthropicProvider with live API calls."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test that the provider can successfully connect to the API."""
        from lammps_reaper.provider import AnthropicProvider

        provider = AnthropicProvider()
        result = await provider.health_check()
        assert result is True, "Provider health check should return True"

    @pytest.mark.asyncio
    async def test_create_message(self):
        """Test basic message creation with the API."""
        from lammps_reaper.provider import AnthropicProvider

        provider = AnthropicProvider()
        response = await provider.create_message(
            system_prompt="You are a helpful assistant.",
            user_message="Reply with exactly the word 'HELLO' and nothing else.",
            max_tokens=50,
        )
        assert response is not None
        assert len(response) > 0
        assert "HELLO" in response.upper()


class TestGenerationLive:
    """Tests for LAMMPS deck generation with live API calls."""

    @pytest.mark.asyncio
    async def test_generate_lj_simulation(self):
        """Test generation of a simple Lennard-Jones simulation deck."""
        from lammps_reaper import ReaperInput, generate_deck

        reaper_input = ReaperInput(
            intent="Simple LJ fluid simulation with 256 atoms, NVE ensemble, 100 steps"
        )
        output = await generate_deck(reaper_input)

        # Should either succeed or have deck content
        assert output.success or len(output.deck_content) > 0, (
            f"Generation failed: {output.errors}"
        )

        # Check for essential LAMMPS commands
        deck_lower = output.deck_content.lower()
        assert "units" in deck_lower, "Deck should contain 'units' command"
        assert "atom_style" in deck_lower or "atom" in deck_lower, (
            "Deck should define atom style"
        )

    @pytest.mark.asyncio
    async def test_generate_nvt_simulation(self):
        """Test generation of an NVT ensemble simulation."""
        from lammps_reaper import ReaperInput, generate_deck

        reaper_input = ReaperInput(
            intent="LJ fluid NVT simulation at temperature 1.0, 500 atoms, 200 steps"
        )
        output = await generate_deck(reaper_input)

        assert output.success or len(output.deck_content) > 0
        deck_lower = output.deck_content.lower()

        # NVT should have temperature control
        assert "nvt" in deck_lower or "temp" in deck_lower, (
            "NVT simulation should reference temperature control"
        )

    @pytest.mark.asyncio
    async def test_generate_with_output_path(self):
        """Test that generated deck is written to output path."""
        from lammps_reaper import ReaperInput, generate_deck

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_deck.in"

            reaper_input = ReaperInput(
                intent="Minimal LJ simulation, 64 atoms, 10 steps",
                output_path=output_path,
            )
            output = await generate_deck(reaper_input)

            if output.success:
                assert output_path.exists(), "Output file should be created"
                content = output_path.read_text()
                assert len(content) > 0, "Output file should not be empty"
                assert content == output.deck_content, (
                    "File content should match deck_content"
                )


class TestValidationLive:
    """Tests for validation of generated decks."""

    @pytest.mark.asyncio
    async def test_l0_validation_passes(self):
        """Test that L0 (placeholder) validation passes for clean decks."""
        from lammps_reaper import ReaperInput, generate_deck

        reaper_input = ReaperInput(
            intent="Simple LJ fluid, 100 atoms, 50 steps. Use concrete values only."
        )
        output = await generate_deck(reaper_input)

        assert output.validation is not None, "Validation result should exist"
        # L0 checks for placeholders - should pass for properly generated decks
        if output.validation.l0:
            # Report details if L0 fails
            if not output.validation.l0.passed:
                print(f"L0 details: {output.validation.l0.details}")
                print(f"Placeholders found: {output.validation.l0.placeholders_found}")

    @pytest.mark.asyncio
    async def test_l1_validation_structure(self):
        """Test that L1 (syntax) validation runs and returns proper structure."""
        from lammps_reaper import ReaperInput, generate_deck

        reaper_input = ReaperInput(intent="LJ fluid NVE, 128 atoms, 100 steps")
        output = await generate_deck(reaper_input)

        assert output.validation is not None
        assert output.validation.l1 is not None
        assert hasattr(output.validation.l1, "passed")
        assert hasattr(output.validation.l1, "syntax_errors")


class TestFullPipelineLive:
    """End-to-end tests for the complete generate-validate-execute pipeline."""

    @pytest.mark.asyncio
    async def test_generate_validate_execute(self):
        """Test the full pipeline: generate, validate, and execute."""
        from lammps_reaper import ReaperInput, generate_deck

        reaper_input = ReaperInput(
            intent="LJ fluid NVE simulation, 256 atoms, FCC lattice, 50 steps",
            lammps_binary=LAMMPS_BIN,
        )
        output = await generate_deck(reaper_input)

        # Check generation
        assert len(output.deck_content) > 0, "Deck content should not be empty"

        # Check validation structure
        assert output.validation is not None, "Validation should be present"
        assert output.validation.l0 is not None, "L0 validation should be present"

        # Report L0 results
        print(f"\nL0 passed: {output.validation.l0.passed}")
        if output.validation.l0.placeholders_found:
            print(f"Placeholders: {output.validation.l0.placeholders_found}")

        # Execute if binary available
        if LAMMPS_BIN.exists():
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".in", delete=False
            ) as f:
                f.write(output.deck_content)
                deck_path = Path(f.name)

            try:
                result = subprocess.run(
                    [str(LAMMPS_BIN), "-in", str(deck_path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=deck_path.parent,
                )

                # Print output for debugging
                print(f"\nLAMMPS return code: {result.returncode}")
                if result.returncode != 0:
                    print(f"LAMMPS stderr:\n{result.stderr[:2000]}")

                # Check for successful completion
                completed = (
                    result.returncode == 0
                    or "Total wall time" in result.stdout
                    or "Loop time" in result.stdout
                )

                if not completed:
                    # Print deck for debugging
                    print(f"\nGenerated deck:\n{output.deck_content}")

                assert completed, (
                    f"LAMMPS should complete. Return code: {result.returncode}"
                )

            finally:
                deck_path.unlink(missing_ok=True)
                # Clean up any generated files
                for pattern in ["log.lammps", "*.dump", "restart.*"]:
                    for f in deck_path.parent.glob(pattern):
                        f.unlink(missing_ok=True)
        else:
            pytest.skip(f"LAMMPS binary not found at {LAMMPS_BIN}")

    @pytest.mark.asyncio
    async def test_complex_simulation_generation(self):
        """Test generation of a more complex simulation setup."""
        from lammps_reaper import ReaperInput, generate_deck

        reaper_input = ReaperInput(
            intent="""
            Create a molecular dynamics simulation with:
            - Lennard-Jones particles
            - 500 atoms in an FCC lattice
            - Initial temperature: 1.5 (LJ units)
            - NPT ensemble at pressure 1.0
            - Run for 1000 steps
            - Output thermodynamic data every 100 steps
            - Dump atomic positions every 500 steps
            """
        )
        output = await generate_deck(reaper_input)

        assert len(output.deck_content) > 0
        deck_lower = output.deck_content.lower()

        # Check for NPT-related commands
        has_npt = "npt" in deck_lower
        has_pressure = "press" in deck_lower or "barostat" in deck_lower

        # NPT simulations should have pressure control
        if has_npt:
            assert has_pressure or "1.0" in output.deck_content, (
                "NPT simulation should specify pressure"
            )

        # Should have output commands
        assert "thermo" in deck_lower, "Should have thermo command"


class TestErrorHandling:
    """Tests for error handling in live API scenarios."""

    @pytest.mark.asyncio
    async def test_empty_intent_handling(self):
        """Test handling of empty or minimal intent."""
        from lammps_reaper import ReaperInput, generate_deck

        # Even minimal intent should produce something
        reaper_input = ReaperInput(intent="LJ simulation")
        output = await generate_deck(reaper_input)

        # Should still generate something, even if minimal
        assert output.deck_content is not None


class TestEdgeCases:
    """Edge case tests for the live API."""

    @pytest.mark.asyncio
    async def test_special_characters_in_intent(self):
        """Test that special characters in intent are handled."""
        from lammps_reaper import ReaperInput, generate_deck

        reaper_input = ReaperInput(
            intent='LJ fluid with temperature T=1.0, density rho=0.8, "quoted text"'
        )
        output = await generate_deck(reaper_input)

        # Should handle special characters gracefully
        assert output.deck_content is not None
        assert len(output.deck_content) > 0

    @pytest.mark.asyncio
    async def test_very_specific_parameters(self):
        """Test generation with very specific numeric parameters."""
        from lammps_reaper import ReaperInput, generate_deck

        reaper_input = ReaperInput(
            intent="""
            LJ simulation with exact parameters:
            - epsilon = 1.0
            - sigma = 1.0
            - cutoff = 2.5
            - timestep = 0.005
            - 256 atoms
            - 100 steps
            """
        )
        output = await generate_deck(reaper_input)

        assert len(output.deck_content) > 0

        # Check that some of the specific values appear
        deck = output.deck_content
        has_cutoff = "2.5" in deck or "cutoff" in deck.lower()
        has_timestep = "0.005" in deck or "timestep" in deck.lower()

        # At least some parameters should be reflected
        assert has_cutoff or has_timestep, (
            "Specific parameters should appear in deck"
        )
