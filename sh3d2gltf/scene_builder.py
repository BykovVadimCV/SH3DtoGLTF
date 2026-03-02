# ============================================================
# scene_builder.py - FULLY UPDATED with semantic naming
# ============================================================

"""
Orchestrates the construction of a complete trimesh.Scene
from parsed SH3D data. Every node gets a semantic name:

  Root
   ├── Level_GroundFloor
   │    ├── Wall_LivingRoom_1
   │    ├── Wall_LivingRoom_2
   │    ├── Wall_Kitchen_1
   │    ├── Floor_LivingRoom
   │    ├── Floor_Kitchen
   │    ├── Ceiling_LivingRoom
   │    ├── Ceiling_Kitchen
   │    ├── Sofa_LivingRoom_1
   │    ├── Table_Kitchen_1
   │    ├── Chair_Kitchen_1
   │    ├── Chair_Kitchen_2
   │    └── Light_LivingRoom_1
   └── Level_FirstFloor
        ├── Wall_Bedroom_1
        ├── Bed_Bedroom_1
        └── ...

Doors and windows are NOT placed as furniture. Instead their
bounding volume is subtracted (boolean difference) from walls
only, producing clean cutouts. Floors and ceilings are never
cut.

All walls are rendered in a uniform neutral grey regardless of
any authored colour / texture in the SH3D file.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from .models import (
    Home, Level, Furniture, Wall, Room, Camera,
    SH3DLightSource,
)
from .archive import SH3DArchive
from .transforms import (
    compute_furniture_transform,
    compute_furniture_transform_simple,
)
from .geometry import (
    create_wall_mesh,
    create_room_floor_mesh,
    create_room_ceiling_mesh,
)
from .materials import MaterialFactory
from .utils import (
    CM_TO_M,
    ModelCache,
    RoomSpatialIndex,
    NodeNameBuilder,
    mat4_identity,
    mat4_translation,
    mat4_rotation_y,
    logger,
)

# ── Visual override constants ─────────────────────────────────
_WALL_GREY: tuple = (179, 179, 179, 255)

# ── Boolean engine preference order ───────────────────────────
_BOOLEAN_ENGINES = ("manifold", "blender", "scad")


class SceneBuilder:
    """
    Builds a trimesh.Scene from an SH3DArchive.

    * Walls        → uniform grey PBR material, with door/window
                     cutouts applied via boolean subtraction.
    * Doors        → pure cutout only; no geometry placed in scene.
    * Windows      → pure cutout only; no geometry placed in scene.
    * Floors       → never cut; placed as-is with authored material.
    * Ceilings     → never cut; placed as-is with authored material.
    * Furniture    → placed normally with authored materials.
    """

    def __init__(
        self,
        archive: SH3DArchive,
        unit_scale: float = CM_TO_M,
        include_cameras: bool = True,
        include_lights: bool = True,
        include_invisible: bool = False,
    ):
        self._archive = archive
        self._home = archive.home
        self._unit_scale = unit_scale
        self._include_cameras = include_cameras
        self._include_lights = include_lights
        self._include_invisible = include_invisible

        self._material_factory = MaterialFactory(archive)
        self._model_cache = ModelCache()
        self._scene = trimesh.Scene()

        self._room_index = RoomSpatialIndex(
            rooms=self._home.rooms,
            levels=self._home.levels,
        )
        self._namer = NodeNameBuilder(self._room_index)

    # ═══════════════════════════════════════════════════════════
    #  PUBLIC
    # ═══════════════════════════════════════════════════════════

    def build(self) -> trimesh.Scene:
        """Build and return the complete scene."""
        logger.info(
            f"Building scene: {len(self._home.levels)} levels, "
            f"{len(self._home.furniture)} furniture, "
            f"{len(self._home.walls)} walls, "
            f"{len(self._home.rooms)} rooms"
        )

        for room in self._home.rooms:
            room_name = self._room_index.get_room_name(room.id)
            logger.debug(
                f"Room: {room_name} (id={room.id}, "
                f"points={len(room.points)}, "
                f"level={room.level_id})"
            )

        if self._home.levels:
            for level in self._home.levels:
                self._process_level(level)
        else:
            self._process_level_items(
                level=None,
                furniture=self._home.furniture,
                walls=self._home.walls,
                rooms=self._home.rooms,
                parent_node="world",
            )

        if self._include_cameras:
            self._add_cameras()

        logger.info(
            f"Scene built: {len(self._scene.geometry)} geometries"
        )
        for name in sorted(self._scene.geometry.keys()):
            logger.debug(f"  Node: {name}")

        return self._scene

    # ═══════════════════════════════════════════════════════════
    #  LEVEL PROCESSING
    # ═══════════════════════════════════════════════════════════

    def _process_level(self, level: Level) -> None:
        if not level.visible and not self._include_invisible:
            logger.debug(f"Skipping invisible level: {level.name}")
            return

        items = self._home.items_on_level(level.id)
        level_node = self._namer.level_name(level)

        logger.info(
            f"Processing {level_node}: "
            f"{len(items['walls'])} walls, "
            f"{len(items['rooms'])} rooms, "
            f"{len(items['furniture'])} furniture"
        )

        self._process_level_items(
            level=level,
            furniture=items["furniture"],
            walls=items["walls"],
            rooms=items["rooms"],
            parent_node=level_node,
        )

    def _process_level_items(
        self,
        level: Optional[Level],
        furniture: List[Furniture],
        walls: List[Wall],
        rooms: List[Room],
        parent_node: str,
    ) -> None:
        level_id = level.id if level else None

        # ── PHASE 1 — Separate doors/windows from regular furniture ──
        openings: List[Furniture] = []
        regular_furniture: List[Furniture] = []

        for piece in furniture:
            if not piece.visible and not self._include_invisible:
                continue
            if not piece.visible_in_3d and not self._include_invisible:
                continue

            if piece.is_door_or_window:
                openings.append(piece)
            else:
                regular_furniture.append(piece)

        logger.debug(
            f"  {len(openings)} openings (door/window), "
            f"{len(regular_furniture)} regular furniture"
        )

        opening_cutters: List[trimesh.Trimesh] = []

        # ── PHASE 2 — Build cutter volumes for every opening ──
        for piece in openings:
            cutter = self._make_opening_cutter(piece, level)
            if cutter is not None:
                opening_cutters.append(cutter)

        # ── PHASE 3 — Build walls, apply cutters, commit ──
        for wall in walls:
            if not wall.visible and not self._include_invisible:
                continue
            entry = self._build_wall_mesh(wall, level, level_id)
            if entry is None:
                continue
            name, mesh, wall_obj = entry
            mesh = self._subtract_cutters(mesh, opening_cutters)
            self._commit_wall(mesh, wall_obj, name, parent_node)

        # ── PHASE 4 — Floors & ceilings (NO cutting) ──
        for room in rooms:
            if not room.visible and not self._include_invisible:
                continue
            self._add_room(room, level, parent_node)

        # ── PHASE 5 — Doors & Windows: visible geometry, wall-fitted ──
        for piece in openings:
            kind = "Window" if self._is_window(piece) else "Door"
            self._add_opening_geometry(piece, level, level_id, parent_node, kind)

        # ── PHASE 6 — Regular furniture ──
        for piece in regular_furniture:
            self._add_furniture(piece, level, level_id, parent_node)

    # ═══════════════════════════════════════════════════════════
    #  CLASSIFICATION HELPERS
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _is_window(piece: Furniture) -> bool:
        """True if the piece is a window rather than a door."""
        combined = (
            (piece.name or "") + " " + (piece.category or "")
        ).lower()
        return "window" in combined

    # ═══════════════════════════════════════════════════════════
    #  WALL PIPELINE  (build → cut → commit)
    # ═══════════════════════════════════════════════════════════

    def _build_wall_mesh(
        self,
        wall: Wall,
        level: Optional[Level],
        level_id: Optional[str],
    ) -> Optional[Tuple[str, trimesh.Trimesh, Wall]]:
        """Create the raw wall mesh. Does NOT add to scene."""
        try:
            mesh = create_wall_mesh(wall, level, self._unit_scale)
            if mesh is None or len(mesh.vertices) == 0:
                return None
            name = self._namer.wall_name(wall, level_id)
            logger.debug(
                f"Wall built: {name} "
                f"({wall.x_start:.0f},{wall.y_start:.0f} → "
                f"{wall.x_end:.0f},{wall.y_end:.0f})"
            )
            return (name, mesh, wall)
        except Exception as e:
            logger.error(f"Failed to build wall {wall.id}: {e}")
            return None

    def _commit_wall(
        self,
        mesh: trimesh.Trimesh,
        wall: Wall,
        name: str,
        parent: str,
    ) -> None:
        """Apply uniform grey material and add wall to scene."""
        try:
            grey_mat = self._material_factory.create_pbr_material(
                name="wall_grey",
                color=_WALL_GREY,
                shininess=0.05,
            )
            mesh.visual = trimesh.visual.TextureVisuals(
                material=grey_mat
            )
        except Exception:
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                face_colors=np.tile(
                    [c / 255.0 for c in _WALL_GREY],
                    (len(mesh.faces), 1),
                ),
            )

        self._scene.add_geometry(
            mesh,
            node_name=name,
            geom_name=name,
            parent_node_name=parent,
        )

    # ═══════════════════════════════════════════════════════════
    #  OPENING CUTTER CONSTRUCTION & CSG
    # ═══════════════════════════════════════════════════════════

    # How far above the nominal floor elevation the door cutter starts.
    # Large enough that the boolean never nicks the floor mesh, yet
    # small enough that the gap is invisible in practice.
    # Windows do NOT use this — they already sit above the floor via
    # their own 'elevation' attribute.
    DOOR_FLOOR_INSET: float = 0.015  # metres (≈ 1.5 cm)

    # Epsilon lift applied to window cutters (and to door cutters on
    # top of DOOR_FLOOR_INSET) to keep the cutter strictly inside the
    # wall volume and away from floor / ceiling planes.
    _CUTTER_EPSILON: float = 0.001   # metres (1 mm)

    def _make_opening_cutter(
            self,
            piece: Furniture,
            level: Optional[Level],
    ) -> Optional[trimesh.Trimesh]:
        """
        Build an axis-aligned box that represents the void a door or
        window punches through one or more walls.

        Sizing rules
        ------------
        Width  : nominal width × 1.02  (1 % pad per side so paper-thin
                 slivers don't survive the boolean)
        Depth  : max(nominal depth × 2.5, 0.60 m)  — generous so the
                 cutter reliably pierces any wall thickness
        Height : computed from *cutter_bottom* to *cutter_top* (see below)

        Bottom-edge rules
        -----------------
        Doors sit at floor level (elevation ≈ 0).  To prevent the cutter
        from intersecting the floor mesh we raise its bottom edge by
        DOOR_FLOOR_INSET (default 1.5 cm).  This leaves a thin band of
        intact wall at the base of each door opening — visually invisible
        but structurally important for clean geometry.

        Windows already have a non-zero elevation attribute in SH3D (the
        sill height), so only the minimal _CUTTER_EPSILON lift is needed
        to keep the cutter strictly above the floor plane.

        Top-edge rule
        -------------
        The top is padded upward by 1 % of the piece height so the lintel
        edge is cleanly removed even when the wall and cutter surfaces are
        nearly co-planar.
        """
        try:
            us = self._unit_scale
            is_door = not self._is_window(piece)
            kind = "Door" if is_door else "Window"

            w = piece.width  * us * 1.02
            d = max(piece.depth * us * 2.5, 0.60)

            level_elevation = level.elevation if level else 0.0

            # Raw bottom of the opening in world-space Y
            raw_bottom = (piece.elevation + level_elevation) * us

            # ── Bottom-edge treatment ────────────────────────────────
            # Doors: raise by the full sill inset so the floor face is
            #        never touched.  The inset is measured from the piece's
            #        own floor level (raw_bottom), not from world origin,
            #        so it works correctly on upper storeys too.
            # Windows: only a tiny epsilon — their elevation is already
            #          above the floor.
            if is_door:
                cutter_bottom = raw_bottom + self.DOOR_FLOOR_INSET
            else:
                cutter_bottom = raw_bottom + self._CUTTER_EPSILON

            # ── Top-edge treatment ───────────────────────────────────
            top_pad = piece.height * us * 0.01   # 1 % upward pad
            cutter_top = (
                (piece.elevation + level_elevation + piece.height) * us
                + top_pad
            )

            h = cutter_top - cutter_bottom

            if h <= 0:
                logger.warning(
                    f"{kind} cutter for '{piece.name}' has non-positive "
                    f"height ({h:.4f} m) after floor inset — skipping."
                )
                return None

            cutter = trimesh.creation.box(extents=[w, h, d])

            # Place the cutter: centre it vertically between bottom and top
            world_y = cutter_bottom + h / 2.0
            world_x = piece.x  * us
            world_z = -piece.y * us

            t_translate = mat4_translation(world_x, world_y, world_z)
            t_yaw      = mat4_rotation_y(-piece.angle)
            cutter.apply_transform(t_translate @ t_yaw)

            logger.debug(
                f"{kind} cutter: '{piece.name}' "
                f"({w:.3f} × {h:.3f} × {d:.3f} m) "
                f"bottom={cutter_bottom:.4f}  top={cutter_top:.4f}  "
                f"inset={'%.4f' % (self.DOOR_FLOOR_INSET if is_door else self._CUTTER_EPSILON)} "
                f"@ ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})"
            )
            return cutter

        except Exception as e:
            logger.warning(
                f"Failed to build opening cutter for "
                f"'{piece.name}': {e}"
            )
            return None

    @staticmethod
    def _aabb_overlap(
        a: trimesh.Trimesh, b: trimesh.Trimesh
    ) -> bool:
        """Fast axis-aligned bounding-box overlap test."""
        if a.bounds is None or b.bounds is None:
            return True
        a0, a1 = a.bounds
        b0, b1 = b.bounds
        return not (
            a1[0] < b0[0] or b1[0] < a0[0] or
            a1[1] < b0[1] or b1[1] < a0[1] or
            a1[2] < b0[2] or b1[2] < a0[2]
        )

    def _subtract_cutters(
        self,
        mesh: trimesh.Trimesh,
        cutters: List[trimesh.Trimesh],
    ) -> trimesh.Trimesh:
        """
        Subtract every overlapping cutter from *mesh*.
        Used ONLY for walls — floors and ceilings are never passed
        through this method.
        """
        if not cutters:
            return mesh

        result = mesh
        for cutter in cutters:
            if not self._aabb_overlap(result, cutter):
                continue
            result = self._boolean_difference(result, cutter)
        return result

    @staticmethod
    def _boolean_difference(
        base: trimesh.Trimesh,
        tool: trimesh.Trimesh,
    ) -> trimesh.Trimesh:
        """
        Single boolean-difference with multi-engine fallback.
        Returns the original *base* if every engine fails.
        """
        for engine in _BOOLEAN_ENGINES:
            try:
                cut = trimesh.boolean.difference(
                    [base, tool], engine=engine
                )
                if cut is not None and len(cut.vertices) > 0:
                    logger.debug(
                        f"Boolean difference OK (engine={engine})"
                    )
                    return cut
            except Exception:
                continue

        logger.warning("Boolean difference failed on all engines")
        return base

    # ═══════════════════════════════════════════════════════════
    #  ROOM  (floor + ceiling — never cut)
    # ═══════════════════════════════════════════════════════════

    def _add_room(
        self,
        room: Room,
        level: Optional[Level],
        parent: str,
    ) -> None:
        # ── Floor (no cutting) ──
        try:
            floor = create_room_floor_mesh(
                room, level, self._unit_scale
            )
            if floor is not None and len(floor.vertices) > 0:
                mat = self._material_factory.create_floor_material(
                    room
                )
                try:
                    floor.visual = trimesh.visual.TextureVisuals(
                        material=mat
                    )
                except Exception:
                    pass

                name = self._namer.floor_name(room)
                logger.debug(f"Floor: {name}")
                self._scene.add_geometry(
                    floor,
                    node_name=name,
                    geom_name=name,
                    parent_node_name=parent,
                )
        except Exception as e:
            logger.warning(
                f"Failed to add floor for room {room.id}: {e}"
            )

        # ── Ceiling (no cutting) ──
        try:
            ceiling = create_room_ceiling_mesh(
                room, level, self._unit_scale
            )
            if ceiling is not None and len(ceiling.vertices) > 0:
                mat = self._material_factory.create_ceiling_material(
                    room
                )
                try:
                    ceiling.visual = trimesh.visual.TextureVisuals(
                        material=mat
                    )
                except Exception:
                    pass

                name = self._namer.ceiling_name(room)
                logger.debug(f"Ceiling: {name}")
                self._scene.add_geometry(
                    ceiling,
                    node_name=name,
                    geom_name=name,
                    parent_node_name=parent,
                )
        except Exception as e:
            logger.warning(
                f"Failed to add ceiling for room {room.id}: {e}"
            )

    # ═══════════════════════════════════════════════════════════
    #  REGULAR FURNITURE
    # ═══════════════════════════════════════════════════════════

    def _add_furniture(
        self,
        piece: Furniture,
        level: Optional[Level],
        level_id: Optional[str],
        parent: str,
    ) -> None:
        # ── Groups ──
        if piece.is_group:
            group_node = self._namer.group_name(piece, level_id)
            logger.debug(
                f"Group: {group_node} "
                f"({len(piece.children)} children)"
            )
            for child in piece.children:
                if child.visible or self._include_invisible:
                    self._add_furniture(
                        child, level, level_id, group_node
                    )
            return

        # ── Load model ──
        model = self._load_furniture_model(piece)

        if model is None:
            logger.warning(
                f"No model for '{piece.name}' "
                f"({piece.model_path}), using placeholder box"
            )
            model = self._create_placeholder(piece)
            transform = compute_furniture_transform_simple(
                piece, level, self._unit_scale
            )
        else:
            transform = compute_furniture_transform(
                piece, model, level, self._unit_scale
            )

        self._material_factory.apply_furniture_materials(
            model, piece
        )

        name = self._namer.furniture_name(piece, level_id)
        logger.debug(
            f"Furniture: {name} ('{piece.name}', "
            f"pos=({piece.x:.0f},{piece.y:.0f}))"
        )

        if isinstance(model, trimesh.Scene):
            for geom_name, geom in model.geometry.items():
                sub_name = f"{name}_part_{geom_name}"
                sub_transform = mat4_identity()
                try:
                    if geom_name in model.graph:
                        _, sub_transform = model.graph[geom_name]
                except Exception:
                    pass
                combined = transform @ sub_transform
                self._scene.add_geometry(
                    geom,
                    node_name=sub_name,
                    geom_name=sub_name,
                    parent_node_name=parent,
                    transform=combined,
                )
        else:
            self._scene.add_geometry(
                model,
                node_name=name,
                geom_name=name,
                parent_node_name=parent,
                transform=transform,
            )

        if piece.is_light and self._include_lights:
            self._add_light_sources(piece, level, name)

    # ═══════════════════════════════════════════════════════════
    #  MODEL LOADING & PLACEHOLDER
    # ═══════════════════════════════════════════════════════════

    def _load_furniture_model(
        self, piece: Furniture
    ) -> Optional[trimesh.Trimesh]:
        if not piece.model_path:
            return None

        if self._model_cache.has(piece.model_path):
            cached = self._model_cache.get(piece.model_path)
            return cached.copy()

        try:
            model_file = self._archive.extract_model(
                piece.model_path
            )
            model = trimesh.load(
                model_file, force="mesh", process=True,
            )
            self._model_cache.put(piece.model_path, model)
            return model.copy()
        except Exception as e:
            logger.warning(
                f"Failed to load model '{piece.model_path}' "
                f"for '{piece.name}': {e}"
            )
            return None

    def _create_placeholder(
        self, piece: Furniture
    ) -> trimesh.Trimesh:
        box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        box.visual = trimesh.visual.ColorVisuals(
            mesh=box,
            face_colors=np.tile(
                [0.6, 0.6, 0.6, 0.8], (len(box.faces), 1)
            ),
        )
        box.metadata["placeholder"] = True
        return box

    # ═══════════════════════════════════════════════════════════
    #  LIGHTS
    # ═══════════════════════════════════════════════════════════

    def _add_light_sources(
        self,
        piece: Furniture,
        level: Optional[Level],
        parent_node: str,
    ) -> None:
        if not piece.light_sources:
            return

        level_elev = level.elevation if level else 0.0

        for i, ls in enumerate(piece.light_sources):
            light_data = {
                "type": "point",
                "name": f"{parent_node}_source{i + 1}",
                "color": [
                    ls.color[0] / 255.0,
                    ls.color[1] / 255.0,
                    ls.color[2] / 255.0,
                ],
                "intensity": piece.power * 100.0,
                "position": [
                    (piece.x + ls.x * piece.width)
                    * self._unit_scale,
                    (piece.elevation + level_elev
                     + ls.y * piece.height)
                    * self._unit_scale,
                    -(piece.y + ls.z * piece.depth)
                    * self._unit_scale,
                ],
            }

            if "lights" not in self._scene.metadata:
                self._scene.metadata["lights"] = []
            self._scene.metadata["lights"].append(light_data)

    # ═══════════════════════════════════════════════════════════
    #  CAMERAS
    # ═══════════════════════════════════════════════════════════

    def _add_cameras(self) -> None:
        if not self._home.cameras:
            return

        camera_data = []
        for i, cam in enumerate(self._home.cameras):
            cam_name = f"Camera_{cam.name or i + 1}"
            camera_data.append({
                "name": cam_name,
                "type": "perspective",
                "position": [
                    cam.x  * self._unit_scale,
                    cam.z  * self._unit_scale,
                    -cam.y * self._unit_scale,
                ],
                "yaw": cam.yaw,
                "pitch": cam.pitch,
                "fov": cam.field_of_view,
                "lens": cam.lens,
            })

        self._scene.metadata["cameras"] = camera_data

    # ═══════════════════════════════════════════════════════════
    #  OPENING GEOMETRY  (doors & windows fitted to wall thickness)
    # ═══════════════════════════════════════════════════════════

    def _add_opening_geometry(
        self,
        piece: Furniture,
        level: Optional[Level],
        level_id: Optional[str],
        parent: str,
        kind: str,   # "Door" or "Window"
    ) -> None:
        """
        Place a door or window mesh scaled so its depth exactly matches
        the host wall's thickness.

        SH3D stores the wall's measured thickness in ``piece.wall_thickness``.
        When that value is present and positive we replace ``piece.depth``
        with it before computing the transform, which causes
        ``compute_furniture_transform`` to scale the model's depth axis to
        the wall thickness rather than the catalogue depth.  Every other
        dimension (width, height) and the full placement/rotation logic are
        unchanged.

        If ``wall_thickness`` is zero or absent (unusual, but possible for
        free-standing door objects) we fall back to the catalogue depth so
        the piece still renders rather than disappearing.
        """
        if not piece.visible and not self._include_invisible:
            return

        try:
            model = self._load_furniture_model(piece)

            # ── Clamp depth to wall thickness ───────────────────────
            # dataclasses.replace creates a shallow copy; the original
            # Furniture object in self._home.furniture is untouched.
            if piece.wall_thickness and piece.wall_thickness > 0:
                if abs(piece.depth - piece.wall_thickness) > 0.1:
                    logger.debug(
                        f"{kind} '{piece.name}': overriding depth "
                        f"{piece.depth:.1f} → wall_thickness "
                        f"{piece.wall_thickness:.1f}"
                    )
                fitted_piece = dataclasses.replace(
                    piece, depth=piece.wall_thickness
                )
            else:
                fitted_piece = piece

            if model is None:
                model = self._create_placeholder(fitted_piece)
                transform = compute_furniture_transform_simple(
                    fitted_piece, level, self._unit_scale
                )
            else:
                transform = compute_furniture_transform(
                    fitted_piece, model, level, self._unit_scale
                )

            self._material_factory.apply_furniture_materials(model, piece)

            if kind == "Window":
                name = self._namer.window_name(piece, level_id)
            else:
                name = self._namer.door_name(piece, level_id)

            logger.debug(
                f"{kind}: {name} ('{piece.name}', "
                f"depth={fitted_piece.depth:.1f}, "
                f"pos=({piece.x:.0f},{piece.y:.0f}))"
            )

            if isinstance(model, trimesh.Scene):
                for geom_name, geom in model.geometry.items():
                    sub_name = f"{name}_part_{geom_name}"
                    sub_transform = mat4_identity()
                    try:
                        if geom_name in model.graph:
                            _, sub_transform = model.graph[geom_name]
                    except Exception:
                        pass
                    self._scene.add_geometry(
                        geom,
                        node_name=sub_name,
                        geom_name=sub_name,
                        parent_node_name=parent,
                        transform=transform @ sub_transform,
                    )
            else:
                self._scene.add_geometry(
                    model,
                    node_name=name,
                    geom_name=name,
                    parent_node_name=parent,
                    transform=transform,
                )

            if piece.is_light and self._include_lights:
                self._add_light_sources(piece, level, name)

        except Exception as e:
            logger.warning(f"Failed to add {kind} '{piece.name}': {e}")