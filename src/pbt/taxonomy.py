"""Public diagnostic categories and the six-group research taxonomy.

Class numbers in manifests are C1..C10; all internal indices are zero based.
These broad dataset categories are not a replacement for an integrated diagnosis.
"""

CLASS_NAMES = (
    "Low-grade glial tumors",
    "High-grade glial tumors",
    "Glioneuronal tumors",
    "High-grade embryonal tumors",
    "Ependymal tumors",
    "Tumors of uncertain histogenesis",
    "Choroid plexus tumors",
    "Pineal tumors",
    "Non-glial tumors",
    "Non-tumor",
)
GROUPS = ((0, 1, 2), (3, 5, 7), (4,), (6,), (8,), (9,))
GROUP_NAMES = ("C1_C2_C3", "C4_C6_C8", "C5", "C7", "C9", "C10")
CLASS_TO_GROUP = tuple(next(g for g, members in enumerate(GROUPS) if c in members) for c in range(10))
LOCATION_NAMES = (
    "Cerebral hemispheric/lobar",
    "Cerebellum",
    "Ventricular/periventricular",
    "Brainstem/tectum",
    "Thalamus/deep gray",
    "Cerebellopontine angle",
    "Posterior fossa, NOS",
    "Pineal region",
    "Sellar/suprasellar/optic",
    "Spinal cord",
    "Unknown",
)
IHC_NAMES = ("GFAP", "Synaptophysin", "INI1", "H3K27M", "ALK1")


def class_index(label):
    if not isinstance(label, str) or label not in {f"C{i}" for i in range(1, 11)}:
        raise ValueError("Labels must be C1 through C10")
    return int(label[1:]) - 1
