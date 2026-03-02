# ============================================================
# models.py - Data classes representing every SH3D entity
# ============================================================

"""
Strongly-typed data classes that mirror the Sweet Home 3D
Home.xml schema. Each class knows how to parse itself from
an lxml Element and how to produce a 4x4 placement matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict, Any
from xml.etree.ElementTree import Element

import numpy as np


# ── Enums ────────────────────────────────────────────────────

class TextureMapping(Enum):
    """How a texture wraps on a surface."""
    PLAN = auto()
    FIT = auto()


class WallSide(Enum):
    LEFT = auto()
    RIGHT = auto()


class LightType(Enum):
    POINT = auto()
    SPOT = auto()
    DIRECTIONAL = auto()
    SUN = auto()


class BasePlanItemType(Enum):
    FURNITURE = "pieceOfFurniture"
    DOOR_OR_WINDOW = "doorOrWindow"
    LIGHT = "light"
    FURNITURE_GROUP = "furnitureGroup"
    WALL = "wall"
    ROOM = "room"
    POLYLINE = "polyline"
    LABEL = "label"
    DIMENSION_LINE = "dimensionLine"


# ── Helper parsing functions ────────────────────────────────

def _float(el: Element, attr: str, default: float = 0.0) -> float:
    val = el.get(attr)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _int(el: Element, attr: str, default: int = 0) -> int:
    val = el.get(attr)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _bool(el: Element, attr: str, default: bool = False) -> bool:
    val = el.get(attr)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _color(el: Element, attr: str) -> Optional[Tuple[int, int, int, int]]:
    """Parse a SH3D color integer (ARGB packed) into (R, G, B, A) 0-255."""
    val = el.get(attr)
    if val is None:
        return None
    try:
        c = int(val)
        # SH3D stores colors as signed 32-bit integers
        if c < 0:
            c += 2**32
        a = (c >> 24) & 0xFF
        r = (c >> 16) & 0xFF
        g = (c >> 8) & 0xFF
        b = c & 0xFF
        # SH3D often stores 0 alpha meaning fully opaque
        if a == 0:
            a = 255
        return (r, g, b, a)
    except (ValueError, TypeError):
        return None


def _parse_model_rotation(el: Element) -> Optional[np.ndarray]:
    """
    Parse the modelRotation attribute.
    SH3D stores it as 9 space-separated floats (row-major 3x3).
    """
    val = el.get("modelRotation")
    if val is None:
        return None
    parts = val.replace(",", " ").split()
    if len(parts) != 9:
        return None
    try:
        floats = [float(p) for p in parts]
        return np.array(floats, dtype=np.float64).reshape(3, 3)
    except (ValueError, TypeError):
        return None


def _parse_model_transformations(el: Element) -> Optional[List[np.ndarray]]:
    """
    Parse per-part deformation matrices stored in
    modelTransformations child elements.
    """
    transforms = []
    for child in el:
        if child.tag == "modelTransformation":
            vals = child.get("matrix")
            if vals:
                parts = vals.replace(",", " ").split()
                if len(parts) == 16:
                    try:
                        mat = np.array(
                            [float(p) for p in parts], dtype=np.float64
                        ).reshape(4, 4)
                        transforms.append(mat)
                    except ValueError:
                        pass
    return transforms if transforms else None


# ── Texture ──────────────────────────────────────────────────

@dataclass
class SH3DTexture:
    """A texture reference inside the .sh3d archive."""
    name: str
    catalog_id: Optional[str] = None
    image_path: str = ""            # path inside ZIP
    width: float = 1.0              # real-world repeat width (cm)
    height: float = 1.0             # real-world repeat height (cm)
    x_offset: float = 0.0
    y_offset: float = 0.0
    angle: float = 0.0              # rotation in radians
    left_to_right_oriented: bool = True
    fitting_area: bool = False

    @classmethod
    def from_element(cls, el: Element) -> Optional[SH3DTexture]:
        if el is None:
            return None
        name = el.get("name", "unnamed_texture")
        image = el.get("image", "")
        catalog_id = el.get("catalogId")
        return cls(
            name=name,
            catalog_id=catalog_id,
            image_path=image,
            width=_float(el, "width", 100.0),
            height=_float(el, "height", 100.0),
            x_offset=_float(el, "xOffset", 0.0),
            y_offset=_float(el, "yOffset", 0.0),
            angle=_float(el, "angle", 0.0),
            left_to_right_oriented=_bool(el, "leftToRightOriented", True),
        )


# ── Material override ───────────────────────────────────────

@dataclass
class SH3DMaterial:
    """Material override applied to a furniture model."""
    name: str
    key: Optional[str] = None       # model material key
    color: Optional[Tuple[int, int, int, int]] = None
    texture: Optional[SH3DTexture] = None
    shininess: float = 0.0
    opacity: float = 1.0
    visible: bool = True

    @classmethod
    def from_element(cls, el: Element) -> Optional[SH3DMaterial]:
        if el is None:
            return None
        name = el.get("name", "")
        key = el.get("key")
        color = _color(el, "color")
        shininess = _float(el, "shininess", 0.0)
        opacity = _float(el, "opacity", 1.0)
        visible = _bool(el, "visible", True)

        texture = None
        tex_el = el.find("texture")
        if tex_el is not None:
            texture = SH3DTexture.from_element(tex_el)

        return cls(
            name=name,
            key=key,
            color=color,
            texture=texture,
            shininess=shininess,
            opacity=opacity,
            visible=visible,
        )


# ── Light source ─────────────────────────────────────────────

@dataclass
class SH3DLightSource:
    """One light emitter within a light-type furniture piece."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    diameter: Optional[float] = None

    @classmethod
    def from_element(cls, el: Element) -> Optional[SH3DLightSource]:
        if el is None:
            return None
        return cls(
            x=_float(el, "x"),
            y=_float(el, "y"),
            z=_float(el, "z"),
            color=_color(el, "color") or (255, 255, 255, 255),
            diameter=_float(el, "diameter", 0.0) or None,
        )


# ── Base plan item (abstract-ish) ───────────────────────────

@dataclass
class PlanItem:
    """Common base for everything that sits on a floor plan."""
    id: str = ""
    name: str = ""
    level_id: Optional[str] = None
    visible: bool = True


# ── Furniture ────────────────────────────────────────────────

@dataclass
class Furniture(PlanItem):
    """
    One piece of furniture, a door/window, or a light fixture.
    Contains everything needed to place the 3D model exactly.
    """
    # ── Catalog reference ──
    catalog_id: Optional[str] = None
    category: str = ""

    # ── Geometry placement ──
    x: float = 0.0                  # plan X (cm)
    y: float = 0.0                  # plan Y (cm) – depth axis
    elevation: float = 0.0          # bottom of piece above level floor (cm)
    width: float = 100.0            # bounding box X (cm)
    depth: float = 100.0            # bounding box Y (cm)
    height: float = 100.0           # bounding box Z (cm)
    angle: float = 0.0              # yaw rotation (radians, CCW from +X)
    roll: float = 0.0
    pitch: float = 0.0
    model_mirrored: bool = False
    back_face_shown: bool = False

    # ── Model / transform ──
    model_path: str = ""            # path inside ZIP (Content digest)
    model_rotation: Optional[np.ndarray] = None  # 3x3
    model_transformations: Optional[List[np.ndarray]] = None
    model_flags: int = 0
    model_size: Optional[float] = None  # original model bounding sphere

    # ── Appearance ──
    color: Optional[Tuple[int, int, int, int]] = None
    texture: Optional[SH3DTexture] = None
    shininess: float = 0.0
    material_overrides: List[SH3DMaterial] = field(default_factory=list)
    icon_path: str = ""

    # ── Door / Window specifics ──
    is_door_or_window: bool = False
    wall_thickness: float = 0.0     # host wall thickness
    wall_distance: float = 0.0
    wall_width: float = 0.0
    wall_left: float = 0.0
    wall_height: float = 0.0
    wall_top: float = 0.0
    cut_out_shape: Optional[str] = None
    wall_cut_out_on_both_sides: bool = False
    bound_to_wall: bool = False

    # ── Light specifics ──
    is_light: bool = False
    light_sources: List[SH3DLightSource] = field(default_factory=list)
    power: float = 0.5

    # ── Group ──
    is_group: bool = False
    children: List[Furniture] = field(default_factory=list)

    # ── Flags ──
    movable: bool = True
    resizable: bool = True
    deformable: bool = True
    horizontally_rotatable: bool = True
    name_visible: bool = False
    visible_in_3d: bool = True

    @classmethod
    def from_element(cls, el: Element) -> Furniture:
        """
        Build a Furniture from an XML element.
        Handles <pieceOfFurniture>, <doorOrWindow>,
        <light>, and <furnitureGroup>.
        """
        tag = el.tag

        obj = cls(
            id=el.get("id", ""),
            name=el.get("name", ""),
            catalog_id=el.get("catalogId"),
            category=el.get("category", ""),
            level_id=el.get("level"),
            visible=_bool(el, "visible", True),

            x=_float(el, "x"),
            y=_float(el, "y"),
            elevation=_float(el, "elevation"),
            width=_float(el, "width", 100.0),
            depth=_float(el, "depth", 100.0),
            height=_float(el, "height", 100.0),
            angle=_float(el, "angle"),
            roll=_float(el, "roll"),
            pitch=_float(el, "pitch"),
            model_mirrored=_bool(el, "modelMirrored"),
            back_face_shown=_bool(el, "backFaceShown"),

            model_path=el.get("model", ""),
            model_rotation=_parse_model_rotation(el),
            model_transformations=_parse_model_transformations(el),
            model_flags=_int(el, "modelFlags"),

            color=_color(el, "color"),
            shininess=_float(el, "shininess"),

            is_door_or_window=(tag == "doorOrWindow"),
            wall_thickness=_float(el, "wallThickness"),
            wall_distance=_float(el, "wallDistance"),
            wall_width=_float(el, "wallWidth"),
            wall_left=_float(el, "wallLeft"),
            wall_height=_float(el, "wallHeight"),
            wall_top=_float(el, "wallTop"),
            cut_out_shape=el.get("cutOutShape"),
            wall_cut_out_on_both_sides=_bool(el, "wallCutOutOnBothSides"),
            bound_to_wall=_bool(el, "boundToWall"),

            is_light=(tag == "light"),
            power=_float(el, "power", 0.5),

            is_group=(tag == "furnitureGroup"),

            movable=_bool(el, "movable", True),
            resizable=_bool(el, "resizable", True),
            deformable=_bool(el, "deformable", True),
            horizontally_rotatable=_bool(
                el, "horizontallyRotatable", True
            ),
            name_visible=_bool(el, "nameVisible"),
            visible_in_3d=_bool(el, "visibleIn3DView", True),
        )

        # ── Texture child element ──
        tex_el = el.find("texture")
        if tex_el is not None:
            obj.texture = SH3DTexture.from_element(tex_el)

        # ── Material overrides ──
        for mat_el in el.findall(".//material"):
            mat = SH3DMaterial.from_element(mat_el)
            if mat is not None:
                obj.material_overrides.append(mat)

        # ── Light sources ──
        for ls_el in el.findall("lightSource"):
            ls = SH3DLightSource.from_element(ls_el)
            if ls is not None:
                obj.light_sources.append(ls)

        # ── Group children (recursive) ──
        if obj.is_group:
            for child_tag in (
                "pieceOfFurniture", "doorOrWindow",
                "light", "furnitureGroup"
            ):
                for child_el in el.findall(child_tag):
                    obj.children.append(Furniture.from_element(child_el))

        return obj


# ── Wall ─────────────────────────────────────────────────────

@dataclass
class Wall(PlanItem):
    """A wall segment defined by start/end points and thickness."""
    x_start: float = 0.0
    y_start: float = 0.0
    x_end: float = 0.0
    y_end: float = 0.0
    height: float = 250.0            # cm
    height_at_end: Optional[float] = None  # for sloped walls
    thickness: float = 10.0          # cm
    arc_extent: Optional[float] = None  # if curved

    # ── Appearance ──
    left_side_color: Optional[Tuple[int, int, int, int]] = None
    right_side_color: Optional[Tuple[int, int, int, int]] = None
    left_side_texture: Optional[SH3DTexture] = None
    right_side_texture: Optional[SH3DTexture] = None
    left_side_shininess: float = 0.0
    right_side_shininess: float = 0.0
    top_color: Optional[Tuple[int, int, int, int]] = None

    # ── Connections ──
    wall_at_start: Optional[str] = None  # id reference
    wall_at_end: Optional[str] = None

    # ── Baseboards ──
    left_side_baseboard_height: float = 0.0
    left_side_baseboard_thickness: float = 0.0
    right_side_baseboard_height: float = 0.0
    right_side_baseboard_thickness: float = 0.0

    pattern: Optional[str] = None

    @classmethod
    def from_element(cls, el: Element) -> Wall:
        obj = cls(
            id=el.get("id", ""),
            name=el.get("name", ""),
            level_id=el.get("level"),
            visible=_bool(el, "visible", True),

            x_start=_float(el, "xStart"),
            y_start=_float(el, "yStart"),
            x_end=_float(el, "xEnd"),
            y_end=_float(el, "yEnd"),
            height=_float(el, "height", 250.0),
            height_at_end=(
                float(el.get("heightAtEnd"))
                if el.get("heightAtEnd") is not None
                else None
            ),
            thickness=_float(el, "thickness", 10.0),
            arc_extent=(
                float(el.get("arcExtent"))
                if el.get("arcExtent") is not None
                else None
            ),

            left_side_color=_color(el, "leftSideColor"),
            right_side_color=_color(el, "rightSideColor"),
            left_side_shininess=_float(el, "leftSideShininess"),
            right_side_shininess=_float(el, "rightSideShininess"),
            top_color=_color(el, "topColor"),

            wall_at_start=el.get("wallAtStart"),
            wall_at_end=el.get("wallAtEnd"),

            pattern=el.get("pattern"),
        )

        # Left / right textures
        for side in ("leftSide", "rightSide"):
            tex_el = el.find(f"{side}Texture")
            if tex_el is None:
                tex_el = el.find(f"{side}Color/../{side}Texture")
            if tex_el is None:
                # Try alternate nesting
                for child in el:
                    if child.tag == f"{side}Texture":
                        tex_el = child
                        break
            if tex_el is not None:
                tex = SH3DTexture.from_element(tex_el)
                if side == "leftSide":
                    obj.left_side_texture = tex
                else:
                    obj.right_side_texture = tex

        return obj


# ── Room ─────────────────────────────────────────────────────

@dataclass
class Room(PlanItem):
    """A room defined by a polygon of floor points."""
    points: List[Tuple[float, float]] = field(default_factory=list)
    area_visible: bool = False
    name_x_offset: float = 0.0
    name_y_offset: float = 0.0

    # ── Floor ──
    floor_visible: bool = True
    floor_color: Optional[Tuple[int, int, int, int]] = None
    floor_texture: Optional[SH3DTexture] = None
    floor_shininess: float = 0.0

    # ── Ceiling ──
    ceiling_visible: bool = True
    ceiling_color: Optional[Tuple[int, int, int, int]] = None
    ceiling_texture: Optional[SH3DTexture] = None
    ceiling_shininess: float = 0.0
    ceiling_flat: bool = True

    @classmethod
    def from_element(cls, el: Element) -> Room:
        obj = cls(
            id=el.get("id", ""),
            name=el.get("name", ""),
            level_id=el.get("level"),
            visible=_bool(el, "visible", True),
            area_visible=_bool(el, "areaVisible"),
            name_x_offset=_float(el, "nameXOffset"),
            name_y_offset=_float(el, "nameYOffset"),
            floor_visible=_bool(el, "floorVisible", True),
            floor_color=_color(el, "floorColor"),
            floor_shininess=_float(el, "floorShininess"),
            ceiling_visible=_bool(el, "ceilingVisible", True),
            ceiling_color=_color(el, "ceilingColor"),
            ceiling_shininess=_float(el, "ceilingShininess"),
            ceiling_flat=_bool(el, "ceilingFlat", True),
        )

        # Points
        for pt_el in el.findall("point"):
            x = _float(pt_el, "x")
            y = _float(pt_el, "y")
            obj.points.append((x, y))

        # Floor / ceiling textures
        ft_el = el.find("floorTexture")
        if ft_el is not None:
            obj.floor_texture = SH3DTexture.from_element(ft_el)
        ct_el = el.find("ceilingTexture")
        if ct_el is not None:
            obj.ceiling_texture = SH3DTexture.from_element(ct_el)

        return obj


# ── Level ────────────────────────────────────────────────────

@dataclass
class Level:
    """One story / level of the home."""
    id: str = ""
    name: str = ""
    elevation: float = 0.0          # cm above ground
    floor_thickness: float = 12.0   # cm
    height: float = 250.0           # floor-to-ceiling cm
    elevation_index: int = 0
    visible: bool = True
    viewable: bool = True
    background_image_visible_on_3d: bool = False

    @classmethod
    def from_element(cls, el: Element) -> Level:
        return cls(
            id=el.get("id", ""),
            name=el.get("name", ""),
            elevation=_float(el, "elevation"),
            floor_thickness=_float(el, "floorThickness", 12.0),
            height=_float(el, "height", 250.0),
            elevation_index=_int(el, "elevationIndex"),
            visible=_bool(el, "visible", True),
            viewable=_bool(el, "viewable", True),
        )


# ── Camera ───────────────────────────────────────────────────

@dataclass
class Camera:
    """Observer or stored camera viewpoint."""
    id: str = ""
    name: str = ""
    attribute: str = ""             # "storedCamera", "observerCamera", etc.
    x: float = 0.0
    y: float = 0.0
    z: float = 170.0                # eye height cm
    yaw: float = 0.0
    pitch: float = 0.0
    field_of_view: float = math.radians(63)
    time: int = 0                   # millisecond timestamp for sun position
    lens: str = "NORMAL"

    @classmethod
    def from_element(cls, el: Element) -> Camera:
        return cls(
            id=el.get("id", ""),
            name=el.get("name", el.tag),
            attribute=el.get("attribute", ""),
            x=_float(el, "x"),
            y=_float(el, "y"),
            z=_float(el, "z", 170.0),
            yaw=_float(el, "yaw"),
            pitch=_float(el, "pitch"),
            field_of_view=_float(el, "fieldOfView", math.radians(63)),
            time=_int(el, "time"),
            lens=el.get("lens", "NORMAL"),
        )


# ── Compass / Environment ───────────────────────────────────

@dataclass
class Compass:
    x: float = 0.0
    y: float = 0.0
    diameter: float = 100.0
    north_direction: float = 0.0    # radians
    longitude: float = 0.0
    latitude: float = 0.0
    time_zone: str = "UTC"
    visible: bool = True

    @classmethod
    def from_element(cls, el: Element) -> Compass:
        return cls(
            x=_float(el, "x"),
            y=_float(el, "y"),
            diameter=_float(el, "diameter", 100.0),
            north_direction=_float(el, "northDirection"),
            longitude=_float(el, "longitude"),
            latitude=_float(el, "latitude"),
            time_zone=el.get("timeZone", "UTC"),
            visible=_bool(el, "visible", True),
        )


@dataclass
class Environment:
    """Rendering / environment settings."""
    ground_color: Optional[Tuple[int, int, int, int]] = None
    ground_texture: Optional[SH3DTexture] = None
    sky_color: Optional[Tuple[int, int, int, int]] = None
    sky_texture: Optional[SH3DTexture] = None
    light_color: Optional[Tuple[int, int, int, int]] = None
    walls_alpha: float = 0.0
    all_levels_visible: bool = False
    ceiling_light_color: Optional[Tuple[int, int, int, int]] = None
    photo_width: int = 400
    photo_height: int = 300
    photo_quality: int = 0
    video_width: int = 320
    video_frame_rate: int = 25
    draw_ceiling_in_3d: bool = True
    subpart_size_limit: float = 0.0

    @classmethod
    def from_element(cls, el: Element) -> Environment:
        obj = cls(
            ground_color=_color(el, "groundColor"),
            sky_color=_color(el, "skyColor"),
            light_color=_color(el, "lightColor"),
            walls_alpha=_float(el, "wallsAlpha"),
            all_levels_visible=_bool(el, "allLevelsVisible"),
            ceiling_light_color=_color(el, "ceilingLightColor"),
            photo_width=_int(el, "photoWidth", 400),
            photo_height=_int(el, "photoHeight", 300),
            draw_ceiling_in_3d=_bool(el, "drawCeilingIn3DView", True),
        )
        gt = el.find("groundTexture")
        if gt is not None:
            obj.ground_texture = SH3DTexture.from_element(gt)
        st = el.find("skyTexture")
        if st is not None:
            obj.sky_texture = SH3DTexture.from_element(st)
        return obj


# ── Full Home ────────────────────────────────────────────────

@dataclass
class Home:
    """
    Complete parsed representation of a Sweet Home 3D project.
    """
    version: str = ""
    name: str = ""
    unit: str = "CENTIMETER"        # CENTIMETER, MILLIMETER, METER, INCH

    levels: List[Level] = field(default_factory=list)
    furniture: List[Furniture] = field(default_factory=list)
    walls: List[Wall] = field(default_factory=list)
    rooms: List[Room] = field(default_factory=list)
    cameras: List[Camera] = field(default_factory=list)
    compass: Optional[Compass] = None
    environment: Optional[Environment] = None

    # Quick lookup maps (built after parsing)
    _levels_by_id: Dict[str, Level] = field(
        default_factory=dict, repr=False
    )
    _walls_by_id: Dict[str, Wall] = field(
        default_factory=dict, repr=False
    )

    def build_indices(self) -> None:
        """Build fast lookup dicts after all lists are populated."""
        self._levels_by_id = {lv.id: lv for lv in self.levels}
        self._walls_by_id = {w.id: w for w in self.walls}

    def get_level(self, level_id: Optional[str]) -> Optional[Level]:
        if level_id is None:
            return None
        return self._levels_by_id.get(level_id)

    def get_wall(self, wall_id: Optional[str]) -> Optional[Wall]:
        if wall_id is None:
            return None
        return self._walls_by_id.get(wall_id)

    def items_on_level(
        self, level_id: Optional[str]
    ) -> Dict[str, list]:
        """Return all items that belong to a given level."""
        return {
            "furniture": [
                f for f in self.furniture if f.level_id == level_id
            ],
            "walls": [
                w for w in self.walls if w.level_id == level_id
            ],
            "rooms": [
                r for r in self.rooms if r.level_id == level_id
            ],
        }