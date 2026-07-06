"""Tests for differentiable road-distance guidance."""

from __future__ import annotations

import torch

from cnr_trajectory.guidance import DistanceFieldRoadGuidance, RoadRasterMetadata


def test_distance_field_guidance_reduces_road_energy() -> None:
    height = width = 21
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    center_col = 10.0
    distance_field = torch.abs(xx.float() - center_col)
    metadata = RoadRasterMetadata(x_min=0.0, x_max=20.0, y_min=0.0, y_max=20.0, y_axis_down=True)
    guidance = DistanceFieldRoadGuidance(
        metadata=metadata,
        distance_field=distance_field,
        guidance_strength=1.0,
        num_guidance_steps=5,
    )

    traj = torch.zeros(2, 8, 2)
    traj[..., 0] = 14.0
    traj[..., 1] = torch.linspace(2.0, 18.0, 8)

    before, before_mean_dist = guidance.road_energy(traj)
    guided, after, mean_dist = guidance(traj)

    assert guided.shape == traj.shape
    assert after < before
    assert mean_dist < before_mean_dist
    assert torch.all(guided[..., 0] < traj[..., 0])


def test_world_to_normalized_grid_y_axis_down_convention() -> None:
    metadata = RoadRasterMetadata(x_min=0.0, x_max=10.0, y_min=0.0, y_max=20.0, y_axis_down=True)
    guidance = DistanceFieldRoadGuidance(metadata=metadata, distance_field=torch.zeros(3, 4))

    traj = torch.tensor([[[0.0, 20.0], [10.0, 0.0], [5.0, 10.0]]])
    grid = guidance.world_to_normalized_grid(traj)

    torch.testing.assert_close(grid[0, 0], torch.tensor([-1.0 + guidance.eps, -1.0 + guidance.eps]))
    torch.testing.assert_close(grid[0, 1], torch.tensor([1.0 - guidance.eps, 1.0 - guidance.eps]))
    torch.testing.assert_close(grid[0, 2], torch.tensor([0.0, 0.0]))


def test_out_of_bounds_coordinates_are_clamped_and_finite() -> None:
    metadata = RoadRasterMetadata(x_min=0.0, x_max=10.0, y_min=0.0, y_max=10.0)
    guidance = DistanceFieldRoadGuidance(metadata=metadata, distance_field=torch.ones(5, 5))
    traj = torch.tensor([[[-100.0, -100.0], [100.0, 100.0]]])

    guided, energy, mean_dist = guidance(traj)

    assert torch.isfinite(guided).all()
    assert torch.isfinite(energy)
    assert torch.isfinite(mean_dist)


def test_should_apply_respects_step_window() -> None:
    metadata = RoadRasterMetadata(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
    guidance = DistanceFieldRoadGuidance(
        metadata=metadata,
        distance_field=torch.zeros(2, 2),
        apply_start_step=3,
        apply_end_step=5,
    )

    assert not guidance.should_apply(2)
    assert guidance.should_apply(3)
    assert guidance.should_apply(5)
    assert not guidance.should_apply(6)
