"""Experiment configuration loaded from YAML files."""

from __future__ import annotations  # Postponed evaluation of type hints.

from collections.abc import Mapping  # Input type hints.
from dataclasses import dataclass, field, replace  # Immutable configs and copy-with-changes.
from pathlib import Path  # Filesystem paths.
from typing import Any  # Loosely typed YAML content.

import yaml  # YAML parsing.

from metricvision.backend.odometry import OdometryConfig  # Back-end settings.

SHARED_MATCHER_PARAMS = (
    "max_keypoints",
    "ratio",
    "mutual_check",
)  # Parameters every matcher accepts; kept when --matcher overrides the config.


@dataclass(frozen=True)
class ComponentConfig:
    """Selects a registered component by name and passes it constructor parameters.

    Attributes:
        name: Registry key (e.g. ``orb``, ``euroc``, ``sgbm``).
        params: Keyword arguments for the component's constructor.
    """

    name: str  # Registry key.
    params: dict[str, Any] = field(default_factory=dict)  # Constructor arguments.

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComponentConfig:
        """Parse ``{"name": ..., "params": {...}}``.

        Args:
            data: Mapping from YAML.

        Returns:
            The parsed component config.

        Raises:
            KeyError: If ``name`` is missing.
        """
        return cls(name=str(data["name"]), params=dict(data.get("params") or {}))  # params is optional.


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to run one (dataset sequence, front end) experiment.

    Attributes:
        dataset: Which dataset loader to use and its arguments.
        matcher: Which front end to test and its arguments.
        depth: Which stereo depth estimator to use (fixed across experiments).
        odometry: Back-end settings (fixed across experiments).
        output_dir: Root folder for results; each run writes to output_dir/<dataset tag>/<run name>.
        max_frames: Optional cap on processed frames (handy for quick tests).
        pipeline: ``"dense"`` (pairwise matcher + SGBM keyframe VO) or ``"sparse"`` (stereo front end, frame-to-frame VO).
        stereo_frontend: Which sparse stereo front end to use when ``pipeline == "sparse"``.
        run_name: Output folder name; defaults to the front-end name.
        prior_depth: Monocular depth network supplying windowed-BA depth priors (sparse pipeline, ``odometry.depth_prior``).
    """

    dataset: ComponentConfig  # Dataset selection.
    matcher: ComponentConfig  # Front-end selection.
    depth: ComponentConfig  # Depth selection.
    odometry: OdometryConfig = field(default_factory=OdometryConfig)  # Back-end settings.
    output_dir: Path = Path("results")  # Results root.
    max_frames: int | None = None  # Frame cap.
    pipeline: str = "dense"  # Which VO pipeline to run.
    stereo_frontend: ComponentConfig = field(default_factory=lambda: ComponentConfig("fast_orb_klt"))  # Sparse front end.
    prior_depth: ComponentConfig = field(default_factory=lambda: ComponentConfig("depth_anything_v2"))  # Monocular depth network for BA priors.
    run_name: str | None = None  # Output folder override.

    def __post_init__(self) -> None:
        """Validate the pipeline name.

        Raises:
            ValueError: If ``pipeline`` is not ``dense`` or ``sparse``.
        """
        if self.pipeline not in ("dense", "sparse"):  # Typo guard.
            raise ValueError(f"pipeline must be 'dense' or 'sparse', got '{self.pipeline}'")  # Clear error.

    @property
    def frontend_name(self) -> str:
        """Return the name of the front end under test.

        Returns:
            Matcher name (dense) or stereo front-end name (sparse).
        """
        return self.matcher.name if self.pipeline == "dense" else self.stereo_frontend.name  # Depends on the pipeline.

    @property
    def name(self) -> str:
        """Return the output folder name of this run.

        Returns:
            ``run_name`` if set, otherwise the front-end name.
        """
        return self.run_name or self.frontend_name  # Explicit name wins.

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        """Build a config from a parsed YAML mapping.

        Args:
            data: Top-level mapping.

        Returns:
            The parsed configuration.
        """
        max_frames = data.get("max_frames")  # Optional frame cap.
        return cls(  # Parse each section.
            dataset=ComponentConfig.from_dict(data["dataset"]),  # Required.
            matcher=ComponentConfig.from_dict(data.get("matcher") or {"name": "orb"}),  # Defaults to ORB.
            depth=ComponentConfig.from_dict(data.get("depth") or {"name": "sgbm"}),  # Defaults to SGBM.
            odometry=OdometryConfig.from_dict(data.get("odometry")),  # Defaults if missing.
            output_dir=Path(data.get("output_dir", "results")),  # Defaults to ./results.
            max_frames=int(max_frames) if max_frames is not None else None,  # None means "all frames".
            pipeline=str(data.get("pipeline", "dense")),  # Dense by default.
            stereo_frontend=ComponentConfig.from_dict(data.get("stereo_frontend") or {"name": "fast_orb_klt"}),  # Sparse front end.
            run_name=data.get("run_name"),  # Optional folder name.
            prior_depth=ComponentConfig.from_dict(data.get("prior_depth") or {"name": "depth_anything_v2"}),  # Prior network.
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load a config from a YAML file.

        Args:
            path: YAML file.

        Returns:
            The parsed configuration.

        Raises:
            ValueError: If the file does not contain a mapping.
        """
        data = yaml.safe_load(Path(path).read_text())  # Parse YAML safely.
        if not isinstance(data, dict):  # Top level must be a mapping.
            raise ValueError(f"{path} must contain a YAML mapping")  # Clear error.
        return cls.from_dict(data)  # Delegate to the dict parser.

    def with_overrides(
        self,
        matcher: str | None = None,
        dataset_root: str | None = None,
        max_frames: int | None = None,
        output_dir: Path | None = None,
        pipeline: str | None = None,
        refine: str | None = None,
        run_name: str | None = None,
        anchor: str | None = None,
        sequence: str | None = None,
        depth: str | None = None,
        window: int | None = None,
        depth_prior: bool | None = None,
        stereo_frontend: str | None = None,
        calibration: str | None = None,
    ) -> ExperimentConfig:
        """Return a copy with command-line overrides applied.

        When the matcher is switched, only parameters shared by every matcher
        (SHARED_MATCHER_PARAMS) are carried over, so the keypoint budget stays
        equal while extractor-specific options are dropped.

        Args:
            matcher: New front-end name.
            dataset_root: New dataset root path.
            max_frames: New frame cap.
            output_dir: New results root.
            pipeline: ``dense`` or ``sparse``.
            refine: PnP refinement method (``none``, ``opencv`` or ``lm``).
            run_name: Output folder name.
            anchor: Sparse-pipeline point anchoring (``previous``, ``birth`` or ``fused``).
            sequence: Dataset sequence id for loaders that take one (KITTI: ``00``-``21``).
            depth: Depth estimator of the dense pipeline; reuses ``prior_depth`` params when the names match.
            window: Windowed-BA size for the sparse pipeline (0 disables).
            depth_prior: Enable/disable monocular depth priors in the windowed BA.
            stereo_frontend: Sparse-pipeline front end (``fast_orb_klt`` classical, ``superpoint_lightglue`` learned);
                its params are kept (the learned front end extends the classical one and accepts them).
            calibration: Rig calibration YAML for datasets that take one (the wheelchair rosbags).

        Returns:
            The updated configuration.
        """
        config = self  # Start from the current config.
        if matcher is not None and matcher != config.matcher.name:  # Switching front end.
            shared = {k: v for k, v in config.matcher.params.items() if k in SHARED_MATCHER_PARAMS}  # Keep fair-comparison params.
            config = replace(config, matcher=ComponentConfig(matcher, shared))  # Swap the matcher.
        if dataset_root is not None:  # Point at a different dataset location.
            config = replace(
                config,
                dataset=ComponentConfig(config.dataset.name, {**config.dataset.params, "root": dataset_root}),
            )  # Replace root.
        if max_frames is not None:  # Limit the run length.
            config = replace(config, max_frames=max_frames)  # Set the cap.
        if output_dir is not None:  # Write results elsewhere.
            config = replace(config, output_dir=output_dir)  # Set the folder.
        if pipeline is not None:  # Switch VO pipeline.
            config = replace(config, pipeline=pipeline)  # Validated in __post_init__.
        if refine is not None:  # Switch pose refinement (for ablations).
            config = replace(config, odometry=replace(config.odometry, pnp=replace(config.odometry.pnp, refine=refine)))  # Nested replace.
        if sequence is not None:  # Pick another sequence of the same dataset.
            config = replace(config, dataset=ComponentConfig(config.dataset.name, {**config.dataset.params, "sequence": sequence}))  # Replace it.
        if depth is not None and depth != config.depth.name:  # Swap the dense pipeline's depth source.
            params = config.prior_depth.params if depth == config.prior_depth.name else {}  # Network settings live in prior_depth.
            config = replace(config, depth=ComponentConfig(depth, dict(params)))  # Replace it.
        if window is not None:  # Windowed BA size.
            config = replace(config, odometry=replace(config.odometry, window_size=window))  # Nested replace.
        if depth_prior is not None:  # Depth priors on/off.
            config = replace(config, odometry=replace(config.odometry, depth_prior=depth_prior))  # Nested replace.
        if anchor is not None:  # Switch sparse point anchoring (for ablations).
            config = replace(config, odometry=replace(config.odometry, sparse_point_anchor=anchor))  # Nested replace.
        if stereo_frontend is not None and stereo_frontend != config.stereo_frontend.name:  # Classical <-> learned.
            config = replace(config, stereo_frontend=ComponentConfig(stereo_frontend, dict(config.stereo_frontend.params)))
        if calibration is not None:  # Another rig file (e.g. a freshly calibrated one for a new recording).
            config = replace(config, dataset=ComponentConfig(config.dataset.name, {**config.dataset.params, "calibration": calibration}))
        if run_name is not None:  # Name the output folder.
            config = replace(config, run_name=run_name)  # Set it.
        return config  # Updated copy (the original is untouched).
