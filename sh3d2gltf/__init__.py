# ============================================================
# __init__.py - Package entry point
# ============================================================

"""
sh3d2gltf - Convert Sweet Home 3D (.sh3d) files to GLTF/GLB.

Preserves furniture placement, scene hierarchy, textures,
materials, lights, cameras, and multi-level support.
"""

__version__ = "1.0.0"

from .converter import convert_sh3d_to_gltf, ConversionOptions
from .archive import SH3DArchive
from .scene_builder import SceneBuilder

__all__ = [
    "convert_sh3d_to_gltf",
    "ConversionOptions",
    "SH3DArchive",
    "SceneBuilder",
]