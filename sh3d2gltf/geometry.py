# ============================================================
# geometry.py - Wall, room, floor, ceiling mesh generation
# ============================================================

"""
Generates triangle meshes for architectural elements:
walls, floors, and ceilings. These don't come from .obj files
inside the archive — they're procedurally created from the
plan geometry stored in Home.xml.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import trimesh

from .models import Wall, Room, Level, SH3DTexture
from .utils import CM_TO_M, logger


def create_wall_mesh(
    wall: Wall,
    level: Optional[Level] = None,
    unit_scale: float = CM_TO_M,
    segments_per_arc: int = 16,
) -> trimesh.Trimesh:
    """
    Create a 3D mesh for a wall segment.

    Walls are defined by start/end points and thickness.
    Curved walls (arc_extent != None) are tessellated.

    The mesh is positioned in GLTF coordinates:
      X → right, Y → up, Z → toward viewer.
    """
    base_elevation = (level.elevation if level else 0.0) * unit_scale
    height_start = wall.height * unit_scale
    height_end = (
        wall.height_at_end
        if wall.height_at_end is not None
        else wall.height
    ) * unit_scale
    thickness = wall.thickness * unit_scale

    if wall.arc_extent is not None and abs(wall.arc_extent) > 1e-6:
        return _create_curved_wall(
            wall, base_elevation, height_start, height_end,
            thickness, unit_scale, segments_per_arc,
        )

    return _create_straight_wall(
        wall, base_elevation, height_start, height_end,
        thickness, unit_scale,
    )


def _create_straight_wall(
    wall: Wall,
    base_elevation: float,
    height_start: float,
    height_end: float,
    thickness: float,
    unit_scale: float,
) -> trimesh.Trimesh:
    """Create geometry for a straight wall segment."""

    # Convert plan coordinates to GLTF
    x1 = wall.x_start * unit_scale
    z1 = -wall.y_start * unit_scale
    x2 = wall.x_end * unit_scale
    z2 = -wall.y_end * unit_scale

    # Wall direction vector and perpendicular
    dx = x2 - x1
    dz = z2 - z1
    length = math.sqrt(dx * dx + dz * dz)
    if length < 1e-10:
        logger.warning(f"Wall {wall.id} has zero length, skipping")
        return trimesh.Trimesh()

    # Unit direction
    ux = dx / length
    uz = dz / length

    # Perpendicular (thickness direction)
    # Right-hand rule: perpendicular is (-uz, ux) in XZ plane
    half_t = thickness / 2.0
    px = -uz * half_t
    pz = ux * half_t

    # 8 corner vertices of the wall box
    # Bottom face
    y_bot = base_elevation
    # Top face (may be sloped)
    y_top_start = base_elevation + height_start
    y_top_end = base_elevation + height_end

    # Vertices: 4 at start cross-section, 4 at end
    # Left side of wall, right side of wall
    vertices = np.array([
        # Start face
        [x1 + px, y_bot,       z1 + pz],   # 0: start, left, bottom
        [x1 - px, y_bot,       z1 - pz],   # 1: start, right, bottom
        [x1 - px, y_top_start, z1 - pz],   # 2: start, right, top
        [x1 + px, y_top_start, z1 + pz],   # 3: start, left, top
        # End face
        [x2 + px, y_bot,       z2 + pz],   # 4: end, left, bottom
        [x2 - px, y_bot,       z2 - pz],   # 5: end, right, bottom
        [x2 - px, y_top_end,   z2 - pz],   # 6: end, right, top
        [x2 + px, y_top_end,   z2 + pz],   # 7: end, left, top
    ], dtype=np.float64)

    # Faces (triangles) — 6 faces of the box, 2 triangles each = 12
    faces = np.array([
        # Left face (looking from outside on the left)
        [0, 4, 7], [0, 7, 3],
        # Right face
        [1, 2, 6], [1, 6, 5],
        # Front face (start)
        [0, 3, 2], [0, 2, 1],
        # Back face (end)
        [4, 5, 6], [4, 6, 7],
        # Top face
        [3, 7, 6], [3, 6, 2],
        # Bottom face
        [0, 1, 5], [0, 5, 4],
    ], dtype=np.int64)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=True,
    )
    mesh.metadata["sh3d_type"] = "wall"
    mesh.metadata["sh3d_id"] = wall.id
    return mesh


def _create_curved_wall(
    wall: Wall,
    base_elevation: float,
    height_start: float,
    height_end: float,
    thickness: float,
    unit_scale: float,
    num_segments: int = 16,
) -> trimesh.Trimesh:
    """Create geometry for a curved (arc) wall segment."""

    x1 = wall.x_start * unit_scale
    z1 = -wall.y_start * unit_scale
    x2 = wall.x_end * unit_scale
    z2 = -wall.y_end * unit_scale

    # Arc center and radius
    mx = (x1 + x2) / 2.0
    mz = (z1 + z2) / 2.0
    dx = x2 - x1
    dz = z2 - z1
    chord = math.sqrt(dx * dx + dz * dz)

    if chord < 1e-10:
        return trimesh.Trimesh()

    half_arc = wall.arc_extent / 2.0
    if abs(math.sin(half_arc)) < 1e-10:
        return _create_straight_wall(
            wall, base_elevation, height_start, height_end,
            thickness, unit_scale,
        )

    radius = chord / (2.0 * abs(math.sin(half_arc)))

    # Direction from midpoint to center
    # Perpendicular to chord
    perp_x = -dz / chord
    perp_z = dx / chord

    sagitta = radius * (1.0 - math.cos(half_arc))
    if wall.arc_extent > 0:
        cx = mx + perp_x * (radius - sagitta)
        cz = mz + perp_z * (radius - sagitta)
    else:
        cx = mx - perp_x * (radius - sagitta)
        cz = mz - perp_z * (radius - sagitta)

    # Start and end angles
    start_angle = math.atan2(z1 - cz, x1 - cx)
    end_angle = math.atan2(z2 - cz, x2 - cx)

    # Generate arc points
    all_vertices = []
    all_faces = []

    for i in range(num_segments + 1):
        t = i / num_segments
        angle = start_angle + t * wall.arc_extent

        # Point on arc centerline
        ax = cx + radius * math.cos(angle)
        az = cz + radius * math.sin(angle)

        # Perpendicular at this point (radial direction)
        nx = math.cos(angle)
        nz = math.sin(angle)
        half_t = thickness / 2.0

        # Height interpolation for sloped walls
        h = height_start + t * (height_end - height_start)
        y_bot = base_elevation
        y_top = base_elevation + h

        # Inner and outer wall surface points
        all_vertices.extend([
            [ax + nx * half_t, y_bot, az + nz * half_t],  # outer bottom
            [ax + nx * half_t, y_top, az + nz * half_t],  # outer top
            [ax - nx * half_t, y_bot, az - nz * half_t],  # inner bottom
            [ax - nx * half_t, y_top, az - nz * half_t],  # inner top
        ])

    # Connect segments with quads (2 triangles each)
    for i in range(num_segments):
        base = i * 4
        nxt = base + 4

        # Outer face
        all_faces.append([base + 0, nxt + 0, nxt + 1])
        all_faces.append([base + 0, nxt + 1, base + 1])

        # Inner face
        all_faces.append([base + 2, base + 3, nxt + 3])
        all_faces.append([base + 2, nxt + 3, nxt + 2])

        # Top face
        all_faces.append([base + 1, nxt + 1, nxt + 3])
        all_faces.append([base + 1, nxt + 3, base + 3])

        # Bottom face
        all_faces.append([base + 0, base + 2, nxt + 2])
        all_faces.append([base + 0, nxt + 2, nxt + 0])

    # Cap start and end
    for cap_base in [0, num_segments * 4]:
        all_faces.append([cap_base + 0, cap_base + 1, cap_base + 3])
        all_faces.append([cap_base + 0, cap_base + 3, cap_base + 2])

    mesh = trimesh.Trimesh(
        vertices=np.array(all_vertices, dtype=np.float64),
        faces=np.array(all_faces, dtype=np.int64),
        process=True,
    )
    mesh.metadata["sh3d_type"] = "wall"
    mesh.metadata["sh3d_id"] = wall.id
    return mesh


def create_room_floor_mesh(
    room: Room,
    level: Optional[Level] = None,
    unit_scale: float = CM_TO_M,
) -> Optional[trimesh.Trimesh]:
    """
    Create a floor mesh from a room polygon.
    The floor sits at the level's elevation.
    """
    if not room.floor_visible or len(room.points) < 3:
        return None

    elevation = (level.elevation if level else 0.0) * unit_scale

    # Convert 2D polygon points to 3D (XZ plane at Y=elevation)
    vertices_2d = np.array(room.points, dtype=np.float64)

    # Create 3D vertices in GLTF coords
    vertices_3d = np.zeros((len(vertices_2d), 3), dtype=np.float64)
    vertices_3d[:, 0] = vertices_2d[:, 0] * unit_scale    # X
    vertices_3d[:, 1] = elevation                           # Y (up)
    vertices_3d[:, 2] = -vertices_2d[:, 1] * unit_scale    # Z = -planY

    # Triangulate the polygon
    try:
        # Use trimesh's path-based triangulation
        from shapely.geometry import Polygon as ShapelyPolygon
        poly = ShapelyPolygon(vertices_2d)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None

        # Earcut triangulation via trimesh
        from trimesh.creation import triangulate_polygon
        tri_vertices, tri_faces = triangulate_polygon(
            poly, engine="earcut"
        )

        # Map triangulated 2D back to 3D
        verts_3d = np.zeros((len(tri_vertices), 3), dtype=np.float64)
        verts_3d[:, 0] = tri_vertices[:, 0] * unit_scale
        verts_3d[:, 1] = elevation
        verts_3d[:, 2] = -tri_vertices[:, 1] * unit_scale

        mesh = trimesh.Trimesh(
            vertices=verts_3d,
            faces=tri_faces,
            process=True,
        )
    except Exception as e:
        logger.warning(
            f"Shapely/earcut triangulation failed for room "
            f"{room.id}: {e}. Falling back to fan triangulation."
        )
        # Fan triangulation fallback (works for convex rooms)
        faces = []
        for i in range(1, len(vertices_3d) - 1):
            faces.append([0, i, i + 1])
        mesh = trimesh.Trimesh(
            vertices=vertices_3d,
            faces=np.array(faces, dtype=np.int64),
            process=True,
        )

    mesh.metadata["sh3d_type"] = "floor"
    mesh.metadata["sh3d_id"] = room.id

    # Generate UV coordinates for texturing
    _apply_planar_uv(mesh, room.floor_texture, axis="y")

    return mesh


def create_room_ceiling_mesh(
    room: Room,
    level: Optional[Level] = None,
    unit_scale: float = CM_TO_M,
) -> Optional[trimesh.Trimesh]:
    """
    Create a ceiling mesh from a room polygon.
    The ceiling sits at level.elevation + level.height.
    """
    if not room.ceiling_visible or len(room.points) < 3:
        return None

    level_elev = (level.elevation if level else 0.0)
    level_height = (level.height if level else 250.0)
    ceiling_y = (level_elev + level_height) * unit_scale

    # Same polygon as floor but at ceiling height, normals flipped
    floor_mesh = create_room_floor_mesh(room, level, unit_scale)
    if floor_mesh is None:
        return None

    # Move to ceiling height and flip normals
    ceiling_verts = floor_mesh.vertices.copy()
    ceiling_verts[:, 1] = ceiling_y

    # Flip face winding to make normals point downward
    ceiling_faces = floor_mesh.faces[:, ::-1].copy()

    mesh = trimesh.Trimesh(
        vertices=ceiling_verts,
        faces=ceiling_faces,
        process=True,
    )
    mesh.metadata["sh3d_type"] = "ceiling"
    mesh.metadata["sh3d_id"] = room.id

    _apply_planar_uv(mesh, room.ceiling_texture, axis="y")

    return mesh


def _apply_planar_uv(
    mesh: trimesh.Trimesh,
    texture: Optional[SH3DTexture],
    axis: str = "y",
) -> None:
    """
    Apply planar UV mapping based on SH3D texture settings.
    Maps XZ world coordinates to UV using the texture's
    real-world repeat dimensions.
    """
    if texture is None or len(mesh.vertices) == 0:
        return

    verts = mesh.vertices
    tex_width = max(texture.width * CM_TO_M, 1e-6)
    tex_height = max(texture.height * CM_TO_M, 1e-6)

    if axis == "y":
        # Floor/ceiling: project onto XZ plane
        u = verts[:, 0] / tex_width
        v = verts[:, 2] / tex_height
    elif axis == "x":
        u = verts[:, 2] / tex_width
        v = verts[:, 1] / tex_height
    else:  # z
        u = verts[:, 0] / tex_width
        v = verts[:, 1] / tex_height

    # Apply texture offset and rotation
    if texture.angle != 0:
        c = math.cos(texture.angle)
        s = math.sin(texture.angle)
        u_new = u * c - v * s
        v_new = u * s + v * c
        u, v = u_new, v_new

    u += texture.x_offset / tex_width
    v += texture.y_offset / tex_height

    # Store UVs — trimesh stores them per-vertex
    uv = np.column_stack([u, v])

    # trimesh expects UV per face-vertex, but for simple cases
    # per-vertex is fine since we have unique vertices per face
    try:
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    except Exception:
        pass