"""
Constituent material database for CFRP micromechanics.

All elastic properties are SI units:
    E  -> Pa
    G  -> Pa
    nu -> dimensionless

The database stores constituent-level properties.
Effective lamina properties are calculated separately by
src/micromechanics.py.
"""

FIBERS = {
    "T300": {
        "E": 230e9,
        "G": 15e9,
        "nu": 0.20,
        "tensile_strength": 3530e6,
        "source": "Literature: T300 constituent elastic properties",
        "source_type": "literature",
        "notes": "Longitudinal constituent properties used for micromechanics.",
    },

    "AS4": {
        "E": 225e9,
        "G": 15e9,
        "nu": 0.20,
        "tensile_strength": 4400e6,
        "source": "Literature: AS4 carbon fiber constituent properties",
        "source_type": "literature",
        "notes": "Constituent elastic properties.",
    },

    "IM7": {
        "E": 276e9,
        "G": 27e9,
        "nu": 0.20,
        "tensile_strength": 5600e6,
        "source": "Literature: IM7 carbon fiber constituent properties",
        "source_type": "literature",
        "notes": "Constituent elastic properties.",
    },

    "E-Glass": {
        "E": 72.4e9,
        "G": 30.0e9,
        "nu": 0.20,
        "tensile_strength": 3450e6,
        "source": "MatWeb: E-Glass Fiber, Generic",
        "source_type": "datasheet/database",
        "notes": "Shear modulus reported as calculated by source.",
    },

    "S-Glass": {
        "E": 86.9e9,
        "G": 35.0e9,
        "nu": 0.22,
        "tensile_strength": 4445e6,
        "source": "MatWeb: S-Glass Fiber, Generic",
        "source_type": "datasheet/database",
        "notes": "Shear modulus reported as calculated by source.",
    },

    "Kevlar 49": {
        "E": 124.79e9,
        "G": 2.62e9,
        "nu": 0.345,
        "tensile_strength": 3000e6,
        "source": "NASA/CR-2003-212352: Kevlar 49 Fiber Properties",
        "source_type": "literature",
        "notes": (
            "Uses NASA simulation constituent dataset. "
            "Shear modulus is directly reported; it is not derived "
            "using an isotropic relation."
        ),
    },
}


MATRICES = {
    "Standard Epoxy": {
        "name": "914C Epoxy",
        "E": 4.0e9,
        "G": 1.481e9,
        "nu": 0.35,
        "source": "Literature: T300/914C constituent properties",
        "source_type": "literature",
        "notes": (
            "Representative standard epoxy system. "
            "914C is used as the specific resin definition."
        ),
    },

    "Toughened Epoxy": {
        "name": "HexPly 8552",
        "E": 4.57e9,
        "G": 1.67e9,
        "nu": 0.37,
        "source": "Zhang et al., Composites Part A, 2015",
        "source_type": "literature",
        "notes": (
            "Representative toughened structural epoxy resin. "
            "8552 is a damage-resistant structural epoxy system."
        ),
    },

    "Vinyl Ester": {
        "name": "Derakane Momentum 411-350",
        "E": 3.2e9,
        "G": 1.230769e9,
        "nu": 0.30,
        "source": "Derakane Momentum 411-350 resin data",
        "source_type": "datasheet/literature",
        "notes": (
            "G was calculated from E/[2(1+nu)] because "
            "the resin source reports E and nu."
        ),
    },
}


def get_fiber(name):
    """Return fiber properties by name."""
    if name not in FIBERS:
        raise KeyError(f"Unknown fiber material: {name}")
    return FIBERS[name]


def get_matrix(name):
    """Return matrix properties by name."""
    if name not in MATRICES:
        raise KeyError(f"Unknown matrix material: {name}")
    return MATRICES[name]


def list_fibers():
    """Return available fiber names."""
    return list(FIBERS.keys())


def list_matrices():
    """Return available matrix names."""
    return list(MATRICES.keys())