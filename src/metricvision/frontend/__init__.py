"""Front ends (feature extraction + matching). Importing this package registers every front end."""

from metricvision.frontend import (  # noqa: F401  # Imported for their registration side effect (torch loads lazily).
    dense_mast3r,
    learned,
    orb,
    sift,
    stereo_klt,
)
from metricvision.frontend.base import MATCHERS, FeatureMatcher, SparseFeatureMatcher  # Pairwise front ends (dense pipeline).
from metricvision.frontend.stereo_klt import STEREO_FRONTENDS, StereoFrontend, StereoTracks  # Sparse stereo front ends.

__all__ = [
    "MATCHERS",
    "STEREO_FRONTENDS",
    "FeatureMatcher",
    "SparseFeatureMatcher",
    "StereoFrontend",
    "StereoTracks",
]  # Names exported by star imports.
