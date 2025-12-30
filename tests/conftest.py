"""Shared fixtures for lammps_reaper tests."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_lammps_deck():
    return """
units lj
atom_style atomic
boundary p p p
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
velocity all create 1.0 87287
fix 1 all nve
thermo 10
run 100
"""


@pytest.fixture
def deck_with_placeholders():
    return "units {{UNITS}}\ntimestep {{TIMESTEP}}"


@pytest.fixture
def deck_with_todo_markers():
    return """units lj
atom_style atomic
# TODO: Add boundary conditions
# FIXME: Check pair_style parameters
"""


@pytest.fixture
def deck_missing_units():
    return """atom_style atomic
boundary p p p
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
"""


@pytest.fixture
def deck_with_bad_syntax():
    return """units lj
atom_style atomic
boundary p p p
print "unbalanced quote
region box block 0 4 0 4 0 4
"""


@pytest.fixture
def deck_with_physics_issues():
    return """units lj
atom_style atomic
boundary p p p
timestep 1.0
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
velocity all create -100.0 87287
fix 1 all nve
run 100
"""


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent.parent / "fixtures"
