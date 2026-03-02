# ============================================================
# transforms.py - Build exact 4×4 matrices for furniture
# ============================================================

"""
Replicates the exact transform logic that Sweet Home 3D uses
when placing a furniture piece in the 3D scene.

The final matrix takes the model from its original OBJ/DAE
coordinate space into GLTF world space, accounting for:
- Position (x, y, elevation) → GLTF (x, elevation, -y)
- Yaw angle (rotation around vertical axis)
- Roll and pitch (if any)
- Model rotation (creator-specified 3×3 matrix)
- Mirroring
- Scale normalization (model bbox → specified width/depth/height)
- Level elevation offset
- Unit conversion (cm → m)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import trimesh

from .models import Furniture, Level
from .utils import (
    CM_TO_M,
    mat4_identity,
    mat4_translation,
    mat4_scale,
    mat4_rotation_y,
    mat4_rotation_x,
    mat4_rotation_z,
    embed_3x3_in_4x4,
    logger,
)


def compute_furniture_transform(
    piece: Furniture,
    model_mesh: trimesh.Trimesh,
    level: Optional[Level] = None,
    unit_scale: float = CM_TO_M,
) -> np.ndarray:
    """
    Compute the full 4×4 world-space transform for a furniture piece.

    Parameters
    ----------
    piece : Furniture
        Parsed furniture data with position, rotation, size, etc.
    model_mesh : trimesh.Trimesh or trimesh.Scene
        The loaded 3D model (used to get its bounding box).
    level : Level, optional
        The level this piece sits on (adds elevation offset).
    unit_scale : float
        Conversion factor from SH3D units to meters.

    Returns
    -------
    np.ndarray
        4×4 homogeneous transformation matrix.
    """

    # ── 1. Get model bounding box ────────────────────────────
    if isinstance(model_mesh, trimesh.Scene):
        bounds = model_mesh.bounds
    else:
        bounds = model_mesh.bounds

    if bounds is None or len(bounds) != 2:
        model_min = np.zeros(3)
        model_max = np.ones(3) * 100.0
    else:
        model_min = bounds[0]
        model_max = bounds[1]

    model_size = model_max - model_min
    # Avoid division by zero
    model_size = np.where(model_size < 1e-10, 1.0, model_size)
    model_center = (model_min + model_max) / 2.0

    # ── 2. Scale to match specified dimensions ───────────────
    # SH3D stores width/depth/height in its own coordinate system:
    #   width  → model X
    #   depth  → model Z (or Y depending on model orientation)
    #   height → model Y (vertical in SH3D's model space)
    #
    # The model might have any orientation, so we use the
    # model_rotation matrix to figure out which bbox axis
    # maps to which dimension.

    sx = (piece.width * unit_scale) / model_size[0]
    sy = (piece.height * unit_scale) / model_size[1]
    sz = (piece.depth * unit_scale) / model_size[2]

    # ── 3. Build the transform chain ─────────────────────────
    # The order matters! We build from inner-most to outer-most:
    #
    #   World = Translation × Yaw × Pitch × Roll × Mirror ×
    #           ModelRotation × Scale × CenterOffset

    # 3a. Center the model at origin
    t_center = mat4_translation(
        -model_center[0], -model_center[1], -model_center[2]
    )

    # 3b. Scale
    t_scale = mat4_scale(sx, sy, sz)

    # 3c. Model rotation (creator-specified, e.g., to orient
    #      a model that was authored lying down)
    if piece.model_rotation is not None:
        t_model_rot = embed_3x3_in_4x4(piece.model_rotation)
    else:
        t_model_rot = mat4_identity()

    # 3d. Mirror
    if piece.model_mirrored:
        t_mirror = mat4_scale(-1.0, 1.0, 1.0)
    else:
        t_mirror = mat4_identity()

    # 3e. Roll / Pitch / Yaw
    # SH3D applies these in the order: roll, pitch, yaw
    # Roll = rotation around model's depth axis (Z in model space)
    # Pitch = rotation around model's width axis (X)
    # Yaw = rotation around vertical (Y)
    t_roll = mat4_rotation_z(piece.roll) if piece.roll != 0 else mat4_identity()
    t_pitch = mat4_rotation_x(piece.pitch) if piece.pitch != 0 else mat4_identity()

    # Yaw: SH3D angle is clockwise when viewed from above
    # In SH3D's coordinate system, angle rotates around Y
    # We negate because SH3D uses CW and we need CCW for standard math
    t_yaw = mat4_rotation_y(-piece.angle)

    # 3f. Translation to world position
    # SH3D coords: (x, y) on plan, elevation above floor.
    # 'elevation' is the BOTTOM of the piece above the level floor.
    # After t_center + t_scale the model spans ±(piece.height*unit_scale/2)
    # in Y, so we must add half the piece height to align the bottom
    # with 'elevation' (same logic as compute_furniture_transform_simple).
    level_elevation = level.elevation if level else 0.0
    world_elevation = (
        (piece.elevation + level_elevation) * unit_scale
        + (piece.height * unit_scale) / 2.0
    )
    world_x = piece.x * unit_scale
    world_z = -piece.y * unit_scale  # SH3D Y → GLTF -Z

    t_translate = mat4_translation(world_x, world_elevation, world_z)

    # ── 4. Compose the full transform ────────────────────────
    # Read right to left: center → scale → model_rot → mirror →
    #                      roll → pitch → yaw → translate
    transform = (
        t_translate
        @ t_yaw
        @ t_pitch
        @ t_roll
        @ t_mirror
        @ t_model_rot
        @ t_scale
        @ t_center
    )

    return transform


def compute_furniture_transform_simple(
    piece: Furniture,
    level: Optional[Level] = None,
    unit_scale: float = CM_TO_M,
) -> np.ndarray:
    """
    Simplified transform when no model mesh is available.
    Uses the piece's own width/depth/height as the box size.
    """
    # Build a unit-cube-like transform
    half_w = piece.width * unit_scale / 2.0
    half_d = piece.depth * unit_scale / 2.0
    half_h = piece.height * unit_scale / 2.0

    level_elevation = level.elevation if level else 0.0
    world_y = (piece.elevation + level_elevation + piece.height / 2.0) * unit_scale
    world_x = piece.x * unit_scale
    world_z = -piece.y * unit_scale

    t_scale = mat4_scale(
        piece.width * unit_scale,
        piece.height * unit_scale,
        piece.depth * unit_scale,
    )

    t_yaw = mat4_rotation_y(-piece.angle)

    if piece.model_mirrored:
        t_mirror = mat4_scale(-1.0, 1.0, 1.0)
    else:
        t_mirror = mat4_identity()

    t_translate = mat4_translation(world_x, world_y, world_z)

    return t_translate @ t_yaw @ t_mirror @ t_scale