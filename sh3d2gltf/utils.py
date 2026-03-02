# ============================================================
# utils.py - UPDATED with room assignment logic
# ============================================================

"""
Shared utilities: coordinate system transforms, model caching,
logging configuration, unit conversions, and room assignment.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger("sh3d2gltf")


# ── Unit conversion ──────────────────────────────────────────

CM_TO_M = 0.01
M_TO_CM = 100.0

UNIT_SCALES: Dict[str, float] = {
    "CENTIMETER": CM_TO_M,
    "MILLIMETER": 0.001,
    "METER": 1.0,
    "INCH": 0.0254,
}


def unit_to_meters(unit: str) -> float:
    return UNIT_SCALES.get(unit.upper(), CM_TO_M)


# ── Coordinate system conversion ────────────────────────────

def sh3d_to_gltf_root_transform(scale: float = CM_TO_M) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[0, 0] = scale
    mat[1, 1] = 0.0
    mat[1, 2] = scale
    mat[2, 1] = -scale
    mat[2, 2] = 0.0
    mat[3, 3] = 1.0
    return mat


def sh3d_position_to_gltf(
    x: float, y: float, elevation: float,
    scale: float = CM_TO_M,
) -> np.ndarray:
    return np.array([
        x * scale,
        elevation * scale,
        -y * scale,
    ], dtype=np.float64)


# ── Model cache ──────────────────────────────────────────────

class ModelCache:
    def __init__(self):
        self._cache: Dict[str, object] = {}
        self._hash_cache: Dict[str, str] = {}

    def get(self, key: str):
        return self._cache.get(key)

    def put(self, key: str, mesh):
        self._cache[key] = mesh

    def has(self, key: str) -> bool:
        return key in self._cache

    def content_hash(self, data: bytes) -> str:
        key = id(data)
        if key not in self._hash_cache:
            self._hash_cache[key] = hashlib.sha256(data).hexdigest()[:16]
        return self._hash_cache[key]

    def clear(self):
        self._cache.clear()
        self._hash_cache.clear()


# ── Color helpers ────────────────────────────────────────────

def rgba_to_float(
    rgba: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[float, float, float, float]]:
    if rgba is None:
        return None
    return (rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, rgba[3] / 255.0)


def color_to_basecolor(
    rgba: Optional[Tuple[int, int, int, int]],
) -> Optional[np.ndarray]:
    if rgba is None:
        return None
    return np.array([
        rgba[0] / 255.0, rgba[1] / 255.0,
        rgba[2] / 255.0, rgba[3] / 255.0,
    ], dtype=np.float64)


def shininess_to_roughness(shininess: float) -> float:
    return max(0.0, min(1.0, 1.0 - shininess))


# ── Matrix utilities ─────────────────────────────────────────

def mat4_identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def mat4_translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def mat4_scale(sx: float, sy: float, sz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def mat4_rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def mat4_rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float64)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def mat4_rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def embed_3x3_in_4x4(rot3: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot3
    return m


# ══════════════════════════════════════════════════════════════
# NEW: Semantic node naming and room assignment
# ══════════════════════════════════════════════════════════════


def sanitize_name(raw: str) -> str:
    """
    Clean a name for use in GLTF node names.
    Remove special characters, replace spaces with underscores,
    collapse multiple underscores.

    'Living Room (2nd floor)' → 'LivingRoom_2ndfloor'
    """
    # Remove characters that are problematic in node names
    cleaned = re.sub(r'[^\w\s-]', '', raw)
    # Replace whitespace with nothing (CamelCase) or underscore
    cleaned = re.sub(r'\s+', '', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned if cleaned else "Unnamed"


class RoomSpatialIndex:
    """
    Spatial index that determines which Room polygon contains
    a given (x, y) point on the SH3D plan.

    Uses Shapely for accurate point-in-polygon tests.
    Falls back to a simple ray-casting algorithm if Shapely
    is not available.
    """

    def __init__(self, rooms: list, levels: list):
        """
        Parameters
        ----------
        rooms : List[Room]
            All rooms from the parsed Home.
        levels : List[Level]
            All levels (used to filter rooms by level).
        """
        self._rooms = rooms
        self._levels = levels

        # Map: room.id → sanitized display name
        self._room_names: Dict[str, str] = {}

        # Map: room.id → Shapely Polygon (or list of points)
        self._room_polygons: Dict[str, object] = {}

        # Map: room.id → level_id
        self._room_levels: Dict[str, Optional[str]] = {}

        # Build the index
        self._build()

    def _build(self) -> None:
        """Build spatial lookup structures."""
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            use_shapely = True
        except ImportError:
            use_shapely = False
            logger.info(
                "Shapely not available — using fallback "
                "point-in-polygon test"
            )

        name_counts: Dict[str, int] = {}

        for room in self._rooms:
            if len(room.points) < 3:
                continue

            # Generate a clean room name
            base_name = sanitize_name(room.name) if room.name else "Room"

            # Handle duplicate room names by appending a counter
            if base_name not in name_counts:
                name_counts[base_name] = 0
            name_counts[base_name] += 1

            if name_counts[base_name] == 1:
                display_name = base_name
            else:
                display_name = f"{base_name}{name_counts[base_name]}"

            self._room_names[room.id] = display_name
            self._room_levels[room.id] = room.level_id

            # Build polygon
            if use_shapely:
                try:
                    poly = ShapelyPolygon(room.points)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    self._room_polygons[room.id] = poly
                except Exception as e:
                    logger.warning(
                        f"Invalid polygon for room '{room.name}': {e}"
                    )
                    self._room_polygons[room.id] = room.points
            else:
                self._room_polygons[room.id] = room.points

        logger.debug(
            f"Room spatial index built: {len(self._room_polygons)} rooms"
        )

    def find_room_for_wall(
        self,
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
        level_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Find the room whose boundary is closest to the wall segment.

        Walls sit *on* room boundaries rather than inside them, so a
        simple point-in-polygon test on the midpoint almost always
        fails.  Instead we sample multiple points along the wall and
        pick the room polygon with the minimum distance to any of those
        sample points.

        Parameters
        ----------
        x_start, y_start, x_end, y_end : float
            Wall endpoint coordinates in SH3D plan space.
        level_id : str, optional
            If provided, only consider rooms on this level.

        Returns
        -------
        str or None
            The room ID of the nearest room, or None if there are no
            rooms in the index.
        """
        best_room_id: Optional[str] = None
        best_dist: float = float("inf")

        for room_id, poly in self._room_polygons.items():
            if level_id is not None:
                if self._room_levels.get(room_id) != level_id:
                    continue

            d = _min_dist_wall_to_polygon(x_start, y_start, x_end, y_end, poly)
            if d < best_dist:
                best_dist = d
                best_room_id = room_id

        return best_room_id

    def get_room_name_for_wall(
        self,
        x_start: float,
        y_start: float,
        x_end: float,
        y_end: float,
        level_id: Optional[str] = None,
    ) -> str:
        """
        Combined lookup: find the nearest room for a wall segment and
        return its sanitized display name.
        """
        room_id = self.find_room_for_wall(
            x_start, y_start, x_end, y_end, level_id
        )
        return self.get_room_name(room_id)

    def find_room_for_opening(
        self,
        x: float,
        y: float,
        level_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Find the room whose boundary is closest to a door or window.

        Doors and windows are placed *on* wall segments which lie on
        room boundaries, so point-in-polygon always misses them.
        We use the same nearest-boundary strategy as for walls.

        Parameters
        ----------
        x, y : float
            SH3D plan coordinates of the door/window centre.
        level_id : str, optional
            If provided, only consider rooms on this level.

        Returns
        -------
        str or None
            The room ID of the nearest room, or None.
        """
        best_room_id: Optional[str] = None
        best_dist: float = float("inf")

        for room_id, poly in self._room_polygons.items():
            if level_id is not None:
                if self._room_levels.get(room_id) != level_id:
                    continue

            d = _min_dist_point_to_polygon(x, y, poly)
            if d < best_dist:
                best_dist = d
                best_room_id = room_id

        return best_room_id

    def get_room_name_for_opening(
        self,
        x: float,
        y: float,
        level_id: Optional[str] = None,
    ) -> str:
        """Combined lookup for a door/window centre point."""
        room_id = self.find_room_for_opening(x, y, level_id)
        return self.get_room_name(room_id)

    def find_room_at(
        self,
        x: float,
        y: float,
        level_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Find which room contains the point (x, y) on the plan.

        Parameters
        ----------
        x, y : float
            SH3D plan coordinates.
        level_id : str, optional
            If provided, only check rooms on this level.

        Returns
        -------
        str or None
            The room ID, or None if the point is outside all rooms.
        """
        for room_id, poly in self._room_polygons.items():
            # Filter by level if specified
            if level_id is not None:
                if self._room_levels.get(room_id) != level_id:
                    continue

            if self._point_in_polygon(x, y, poly):
                return room_id

        return None

    def get_room_name(self, room_id: Optional[str]) -> str:
        """
        Get the sanitized display name for a room.
        Returns 'None' if room_id is None (item not in any room).
        """
        if room_id is None:
            return "Common"
        return self._room_names.get(room_id, "Unknown")

    def get_room_name_at(
        self,
        x: float,
        y: float,
        level_id: Optional[str] = None,
    ) -> str:
        """
        Combined lookup: find room at (x,y) and return its name.
        """
        room_id = self.find_room_at(x, y, level_id)
        return self.get_room_name(room_id)

    @staticmethod
    def _point_in_polygon(
        x: float, y: float, polygon
    ) -> bool:
        """
        Test if point (x, y) is inside the polygon.
        Uses Shapely if polygon is a Shapely object,
        otherwise falls back to ray casting.
        """
        try:
            # Shapely Polygon
            from shapely.geometry import Point
            return polygon.contains(Point(x, y))
        except (ImportError, AttributeError, TypeError):
            pass

        # Fallback: ray-casting algorithm
        if isinstance(polygon, (list, tuple)):
            return _ray_cast_contains(x, y, polygon)
        return False


def _ray_cast_contains(
    px: float, py: float,
    polygon: List[Tuple[float, float]],
) -> bool:
    """
    Ray-casting point-in-polygon test.
    Works without Shapely.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _wall_midpoint(wall) -> Tuple[float, float]:
    """Get the midpoint of a wall segment in plan coordinates."""
    return (
        (wall.x_start + wall.x_end) / 2.0,
        (wall.y_start + wall.y_end) / 2.0,
    )


def _dist_point_to_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Euclidean distance from point (px,py) to segment (a→b)."""
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return ((px - ax - t * dx) ** 2 + (py - ay - t * dy) ** 2) ** 0.5


def _min_dist_point_to_polygon(
    px: float, py: float,
    polygon,
) -> float:
    """
    Minimum distance from a point to any edge of a polygon.

    Works with both Shapely polygons (exterior boundary) and
    plain lists of (x, y) tuples.
    """
    try:
        # Shapely path
        from shapely.geometry import Point
        return polygon.exterior.distance(Point(px, py))
    except (AttributeError, ImportError):
        pass

    # Fallback: iterate over edges
    if not isinstance(polygon, (list, tuple)):
        return float("inf")
    n = len(polygon)
    if n == 0:
        return float("inf")
    min_d = float("inf")
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        d = _dist_point_to_segment(px, py, ax, ay, bx, by)
        if d < min_d:
            min_d = d
    return min_d


def _min_dist_wall_to_polygon(
    x1: float, y1: float,
    x2: float, y2: float,
    polygon,
    samples: int = 7,
) -> float:
    """
    Minimum distance from a wall segment (sampled at *samples+1*
    evenly-spaced points) to the boundary of *polygon*.

    Sampling covers the wall midpoint as well as both endpoints,
    catching walls that run along a room edge without crossing it.
    """
    min_d = float("inf")
    for i in range(samples + 1):
        t = i / samples
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        d = _min_dist_point_to_polygon(px, py, polygon)
        if d < min_d:
            min_d = d
    return min_d


# ── Node name builder ───────────────────────────────────────

class NodeNameBuilder:
    """
    Generates unique, semantic node names following the pattern:

        {Type}_{RoomName}_{InstanceNumber}

    Tracks instance counts per (type, room) pair to ensure
    every node name in the GLTF is unique.
    """

    def __init__(self, room_index: RoomSpatialIndex):
        self._room_index = room_index
        # Counts per full prefix, e.g. "Wall_Room1" → 3
        self._counts: Dict[str, int] = {}

    def wall_name(
        self, wall, level_id: Optional[str] = None
    ) -> str:
        """
        Generate name for a wall node.

        Walls sit on room boundaries, so a midpoint-in-polygon test
        almost always fails.  Instead we find the room whose boundary
        polygon is geometrically closest to the wall segment.
        """
        room_name = self._room_index.get_room_name_for_wall(
            wall.x_start, wall.y_start,
            wall.x_end,   wall.y_end,
            level_id,
        )
        return self._next_name("Wall", room_name)

    def door_name(
        self, piece, level_id: Optional[str] = None
    ) -> str:
        """Generate name for a door node using nearest-boundary lookup."""
        room_name = self._room_index.get_room_name_for_opening(
            piece.x, piece.y, level_id
        )
        return self._next_name("Door", room_name)

    def window_name(
        self, piece, level_id: Optional[str] = None
    ) -> str:
        """Generate name for a window node using nearest-boundary lookup."""
        room_name = self._room_index.get_room_name_for_opening(
            piece.x, piece.y, level_id
        )
        return self._next_name("Window", room_name)

    def furniture_name(
        self,
        piece,
        level_id: Optional[str] = None,
        custom_type: Optional[str] = None,
    ) -> str:
        """
        Generate name for a furniture node.
        Uses piece.name or category to determine a more
        specific type if possible.
        """
        # Determine the type label
        if custom_type:
            type_label = custom_type
        elif piece.is_light:
            type_label = "Light"
        elif piece.is_door_or_window:
            # Further classify as Door or Window
            type_label = self._classify_door_or_window(piece)
        else:
            type_label = self._classify_furniture(piece)

        room_name = self._room_index.get_room_name_at(
            piece.x, piece.y, level_id
        )
        return self._next_name(type_label, room_name)

    def floor_name(self, room) -> str:
        """Generate name for a floor node."""
        room_name = self._room_index.get_room_name(room.id)
        # Floors are unique per room — no instance counter needed
        name = f"Floor_{room_name}"
        if name in self._counts:
            self._counts[name] += 1
            return f"{name}_{self._counts[name]}"
        self._counts[name] = 0
        return name

    def ceiling_name(self, room) -> str:
        """Generate name for a ceiling node."""
        room_name = self._room_index.get_room_name(room.id)
        name = f"Ceiling_{room_name}"
        if name in self._counts:
            self._counts[name] += 1
            return f"{name}_{self._counts[name]}"
        self._counts[name] = 0
        return name

    def group_name(
        self, piece, level_id: Optional[str] = None
    ) -> str:
        """Generate name for a furniture group node."""
        room_name = self._room_index.get_room_name_at(
            piece.x, piece.y, level_id
        )
        return self._next_name("Group", room_name)

    def level_name(self, level) -> str:
        """Generate name for a level parent node."""
        clean = sanitize_name(level.name) if level.name else f"Level{level.elevation_index}"
        return f"Level_{clean}"

    def _next_name(self, type_label: str, room_name: str) -> str:
        """
        Generate the next unique name for a (type, room) pair.

        First instance:  Wall_Room1_1
        Second instance: Wall_Room1_2
        """
        prefix = f"{type_label}_{room_name}"
        if prefix not in self._counts:
            self._counts[prefix] = 0
        self._counts[prefix] += 1
        return f"{prefix}_{self._counts[prefix]}"

    @staticmethod
    def _classify_door_or_window(piece) -> str:
        """
        Determine if a doorOrWindow piece is a Door or Window
        based on its name, category, or catalog ID.
        """
        searchable = (
            (piece.name or "")
            + " "
            + (piece.category or "")
            + " "
            + (piece.catalog_id or "")
        ).lower()

        if "door" in searchable:
            return "Door"
        if "window" in searchable:
            return "Window"
        if "gate" in searchable:
            return "Door"
        if "opening" in searchable:
            return "Opening"

        # Default: check SH3D category conventions
        # Doors typically have larger depth relative to height
        return "DoorOrWindow"

    @staticmethod
    def _classify_furniture(piece) -> str:
        """
        Optionally provide a more specific type label based
        on the furniture's name or category.

        Returns 'Furniture' if no specific match is found.
        """
        searchable = (
            (piece.name or "")
            + " "
            + (piece.category or "")
        ).lower()

        # Common furniture categories
        categories = {
            "table": "Table",
            "desk": "Desk",
            "chair": "Chair",
            "sofa": "Sofa",
            "couch": "Sofa",
            "bed": "Bed",
            "shelf": "Shelf",
            "bookcase": "Shelf",
            "cabinet": "Cabinet",
            "wardrobe": "Wardrobe",
            "closet": "Wardrobe",
            "sink": "Sink",
            "toilet": "Toilet",
            "bathtub": "Bathtub",
            "shower": "Shower",
            "stove": "Stove",
            "oven": "Oven",
            "fridge": "Fridge",
            "refrigerator": "Fridge",
            "washer": "Washer",
            "dryer": "Dryer",
            "lamp": "Lamp",
            "plant": "Plant",
            "rug": "Rug",
            "carpet": "Rug",
            "mirror": "Mirror",
            "tv": "TV",
            "television": "TV",
            "stairs": "Stairs",
            "staircase": "Stairs",
            "railing": "Railing",
            "column": "Column",
            "pillar": "Column",
        }

        for keyword, label in categories.items():
            if keyword in searchable:
                return label

        return "Furniture"