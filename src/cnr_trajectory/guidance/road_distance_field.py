"""Differentiable road-distance guidance for trajectory diffusion sampling.

The guidance operates in continuous trajectory coordinate space. A DDPM can
still generate GASF images; decode the sample to ``[B, T, 2]`` trajectories,
apply this guidance module, and encode the guided trajectories back to GASF.
No DDPM architecture or training code needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RoadRasterMetadata:
    """World/raster conversion metadata for a road mask or distance field.

    ``x_min, x_max, y_min, y_max`` describe the world-coordinate bounding box
    covered by the raster. The tensor/image is indexed as ``[row, col]``.

    By default ``y_axis_down=True``, matching common image/geospatial rasters:
    row 0 corresponds to ``y_max`` and the last row corresponds to ``y_min``.
    In PyTorch ``grid_sample``, normalized y = -1 samples the top row and
    normalized y = +1 samples the bottom row, so this flag controls whether
    world y must be flipped before sampling.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    y_axis_down: bool = True

    @classmethod
    def from_origin(
        cls,
        *,
        x_origin: float,
        y_origin: float,
        pixel_size_x: float,
        pixel_size_y: float,
        width: int,
        height: int,
        y_axis_down: bool = True,
    ) -> "RoadRasterMetadata":
        """Create metadata from a top-left origin and pixel sizes.

        This is convenient for image-style rasters. With ``y_axis_down=True``,
        ``y_origin`` is the world y at the top of the raster.
        """

        x_max = x_origin + float(width) * float(pixel_size_x)
        if y_axis_down:
            y_min = y_origin - float(height) * abs(float(pixel_size_y))
            y_max = y_origin
        else:
            y_min = y_origin
            y_max = y_origin + float(height) * float(pixel_size_y)
        return cls(
            x_min=float(x_origin),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
            y_axis_down=y_axis_down,
        )


class DistanceFieldRoadGuidance(torch.nn.Module):
    """Gradient guidance that pulls trajectories toward a road distance field.

    Parameters
    ----------
    metadata:
        World-coordinate extent and y-axis convention for the raster.
    distance_field:
        Precomputed distance-to-road raster with shape ``[H, W]``. Values
        should be in world distance units, for example meters.
    road_mask:
        Binary road raster with shape ``[H, W]`` where roads are 1 and
        non-roads are 0. If ``distance_field`` is not supplied, this is
        converted to a distance field using SciPy's Euclidean distance
        transform.
    guidance_strength:
        Gradient-descent step size ``eta`` for each guidance update.
    num_guidance_steps:
        Number of inner gradient steps applied each time ``forward`` is called.
    apply_start_step, apply_end_step:
        Inclusive step-index window for ``should_apply(step_idx)``.
    distance_scale:
        Divisor applied to sampled distances inside the squared energy. Use
        this to keep gradients well-scaled if the distance field is in meters
        and values are large. Returned ``mean_road_dist`` remains unscaled.
    clamp_normalized_grid:
        Clamp normalized sampling coordinates to the raster extent. This keeps
        out-of-bounds samples finite and avoids sampling undefined map regions,
        but clamp saturation also means far outside points may receive little
        or no gradient pulling them back into the raster. Disable this only if
        the caller intentionally wants ``grid_sample`` padding behavior to
        shape out-of-bounds gradients.
    """

    def __init__(
        self,
        *,
        metadata: RoadRasterMetadata,
        distance_field: torch.Tensor | np.ndarray | str | Path | None = None,
        road_mask: torch.Tensor | np.ndarray | str | Path | None = None,
        pixel_size: float | tuple[float, float] | None = None,
        guidance_strength: float = 1.0,
        num_guidance_steps: int = 1,
        apply_start_step: int = 0,
        apply_end_step: int | None = None,
        distance_scale: float = 1.0,
        padding_mode: Literal["border", "zeros", "reflection"] = "border",
        align_corners: bool = True,
        clamp_normalized_grid: bool = True,
        eps: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if distance_field is None and road_mask is None:
            raise ValueError("Provide either distance_field or road_mask.")
        if distance_field is not None and road_mask is not None:
            raise ValueError("Provide only one of distance_field or road_mask, not both.")
        if num_guidance_steps < 0:
            raise ValueError(f"num_guidance_steps must be non-negative, got {num_guidance_steps}")
        if distance_scale <= 0:
            raise ValueError(f"distance_scale must be positive, got {distance_scale}")

        self.metadata = metadata
        self.guidance_strength = float(guidance_strength)
        self.num_guidance_steps = int(num_guidance_steps)
        self.apply_start_step = int(apply_start_step)
        self.apply_end_step = None if apply_end_step is None else int(apply_end_step)
        self.distance_scale = float(distance_scale)
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)
        self.clamp_normalized_grid = bool(clamp_normalized_grid)
        self.eps = float(eps)

        field = (
            self._load_array(distance_field)
            if distance_field is not None
            else self._distance_field_from_mask(self._load_array(road_mask), pixel_size)
        )
        field_tensor = torch.as_tensor(field, dtype=dtype, device=device)
        if field_tensor.ndim != 2:
            raise ValueError(f"Distance field must have shape [H, W], got {tuple(field_tensor.shape)}")
        if not torch.isfinite(field_tensor).all():
            raise ValueError("Distance field contains non-finite values.")

        self.height = int(field_tensor.shape[0])
        self.width = int(field_tensor.shape[1])
        self.register_buffer("distance_field", field_tensor[None, None], persistent=True)

    def should_apply(self, step_idx: int) -> bool:
        """Return whether guidance should be applied at this sampling step."""

        if step_idx < self.apply_start_step:
            return False
        if self.apply_end_step is not None and step_idx > self.apply_end_step:
            return False
        return self.num_guidance_steps > 0 and self.guidance_strength != 0.0

    def forward(self, traj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply road guidance to trajectories.

        Parameters
        ----------
        traj:
            Continuous world-coordinate trajectories with shape ``[B, T, 2]``.

        Returns
        -------
        guided_traj:
            Guided trajectories with shape ``[B, T, 2]``.
        road_energy:
            Mean squared sampled road distance after the final guidance step.
        mean_road_dist:
            Mean sampled distance-to-road after the final guidance step.
        """

        if traj.ndim != 3 or traj.shape[-1] != 2:
            raise ValueError(f"Expected traj shape [B, T, 2], got {tuple(traj.shape)}")

        guided = traj.detach().to(device=self.distance_field.device, dtype=self.distance_field.dtype)
        for _ in range(self.num_guidance_steps):
            guided = guided.detach().requires_grad_(True)
            energy, _ = self.road_energy(guided)
            (grad,) = torch.autograd.grad(energy, guided, create_graph=False)
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            guided = guided - self.guidance_strength * grad
            guided = torch.nan_to_num(guided.detach(), nan=0.0, posinf=0.0, neginf=0.0)

        final_energy, final_dist = self.road_energy(guided)
        return guided.to(device=traj.device, dtype=traj.dtype), final_energy.detach(), final_dist.detach()

    def road_energy(self, traj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``E_road = mean(D(x_i, y_i)^2)`` and mean distance."""

        distances = self.sample_distance(traj)
        scaled = distances / self.distance_scale
        energy = torch.mean(scaled.square())
        mean_distance = torch.mean(distances)
        return energy, mean_distance

    def sample_distance(self, traj: torch.Tensor) -> torch.Tensor:
        """Sample distance values at continuous world-coordinate positions.

        ``grid_sample`` expects a grid with last dimension ``(x_norm, y_norm)``
        in ``[-1, 1]``. The output has shape ``[B, T]``.
        """

        grid = self.world_to_normalized_grid(traj).view(traj.shape[0], traj.shape[1], 1, 2)
        field = self.distance_field.expand(traj.shape[0], -1, -1, -1)
        sampled = F.grid_sample(
            field,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return sampled[:, 0, :, 0]

    def world_to_normalized_grid(self, traj: torch.Tensor) -> torch.Tensor:
        """Convert world ``(x, y)`` to ``grid_sample`` normalized coordinates."""

        x = traj[..., 0]
        y = traj[..., 1]
        x_den = max(self.metadata.x_max - self.metadata.x_min, self.eps)
        y_den = max(self.metadata.y_max - self.metadata.y_min, self.eps)

        x_norm = 2.0 * (x - self.metadata.x_min) / x_den - 1.0
        y_fraction = (y - self.metadata.y_min) / y_den
        if self.metadata.y_axis_down:
            y_norm = 1.0 - 2.0 * y_fraction
        else:
            y_norm = 2.0 * y_fraction - 1.0

        grid = torch.stack([x_norm, y_norm], dim=-1)
        if self.clamp_normalized_grid:
            limit = 1.0 - self.eps
            grid = torch.clamp(grid, min=-limit, max=limit)
        return grid

    @staticmethod
    def _load_array(value: torch.Tensor | np.ndarray | str | Path | None) -> torch.Tensor | np.ndarray:
        if value is None:
            raise ValueError("Expected an array or path, got None.")
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.suffix == ".npy":
                return np.load(path)
            raise ValueError(f"Unsupported raster file extension {path.suffix!r}; use .npy or pass an array.")
        return value

    @staticmethod
    def _distance_field_from_mask(
        road_mask: torch.Tensor | np.ndarray,
        pixel_size: float | tuple[float, float] | None,
    ) -> np.ndarray:
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError as exc:
            raise ImportError("Computing a distance field from road_mask requires scipy.") from exc

        mask = np.asarray(road_mask.detach().cpu() if isinstance(road_mask, torch.Tensor) else road_mask)
        if mask.ndim != 2:
            raise ValueError(f"road_mask must have shape [H, W], got {mask.shape}")
        road = mask.astype(bool)
        non_road = ~road
        sampling = None
        if pixel_size is not None:
            if isinstance(pixel_size, tuple):
                pixel_size_x, pixel_size_y = pixel_size
                sampling = (float(pixel_size_y), float(pixel_size_x))
            else:
                sampling = float(pixel_size)
        return distance_transform_edt(non_road, sampling=sampling).astype(np.float32)


DDPM_REVERSE_LOOP_EXAMPLE = """
for step_idx, t in enumerate(reversed(timesteps)):
    x = ddpm_denoise_step(x, t)

    if guidance.should_apply(step_idx):
        traj = gasf_decode(x)              # [B, T, 2], world coordinates
        traj, road_energy, mean_road_dist = guidance(traj)
        x = gasf_encode(traj)
"""
