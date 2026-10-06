"""
Verified unidirectional lamina strength database.

All strengths are SI units:
    Pa

Important:
These are composite-system/lamina properties, NOT universal fiber properties.
Only material systems with a documented strength dataset are included.
"""

LAMINA_STRENGTHS = {
    "T300_BSL914C": {
        "fiber": "T300",
        "matrix": "Standard Epoxy",
        "system_name": "T300/BSL914C",

        "Xt": 1500e6,
        "Xc": 900e6,
        "Yt": 27e6,
        "Yc": 200e6,
        "S": 80e6,

        "source": "NPTEL, Lecture 21: Strength parameters for composite systems",
        "source_type": "literature",
        "notes": (
            "Unidirectional lamina strength dataset for T300/BSL914C."
        ),
    },

    "AS4_3501_6": {
        "fiber": "AS4",
        "matrix": "Standard Epoxy",
        "system_name": "AS4/3501-6",

        "Xt": 1950e6,
        "Xc": 1480e6,
        "Yt": 48e6,
        "Yc": 200e6,
        "S": 79e6,

        "source": "NPTEL, Lecture 21: Strength parameters for composite systems",
        "source_type": "literature",
        "notes": (
            "Unidirectional lamina strength dataset for AS4/3501-6."
        ),
    },

    "IM7_8552": {
        "fiber": "IM7",
        "matrix": "Toughened Epoxy",
        "system_name": "IM7/8552",

        "Xt": 2326.2e6,
        "Xc": 1200.1e6,
        "Yt": 62.3e6,
        "Yc": 199.8e6,
        "S": 92.3e6,

        "source": (
            "Published IM7/8552 unidirectional composite strength dataset"
        ),
        "source_type": "literature",
        "notes": (
            "UD lamina strengths. Values are intended for the current "
            "first-ply-failure/failure-criterion implementation."
        ),
    },
}


def get_lamina_strengths(system_id):
    """
    Return the strength dataset for a composite material system.
    """
    if system_id not in LAMINA_STRENGTHS:
        raise KeyError(f"Unknown lamina strength system: {system_id}")

    return LAMINA_STRENGTHS[system_id]


def find_lamina_strengths(fiber, matrix):
    """
    Find a verified strength dataset matching the selected
    fiber and matrix.
    """
    for system_id, data in LAMINA_STRENGTHS.items():
        if data["fiber"] == fiber and data["matrix"] == matrix:
            return {
                "system_id": system_id,
                **data,
            }

    return None


def list_lamina_strengths():
    """
    Return all available verified composite-system strength datasets.
    """
    return list(LAMINA_STRENGTHS.keys())