# ============================================================
# materials.py - Texture and material handling
# ============================================================

"""
Loads textures from the .sh3d archive, creates trimesh
material objects, and applies color/texture/shininess overrides
from the SH3D material system.
"""

from __future__ import annotations

import io
from typing import Dict, Optional, Tuple, List

import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial, SimpleMaterial

from .models import Furniture, SH3DTexture, SH3DMaterial, Wall, Room
from .archive import SH3DArchive
from .utils import (
    rgba_to_float,
    color_to_basecolor,
    shininess_to_roughness,
    CM_TO_M,
    logger,
)


class MaterialFactory:
    """
    Creates and caches trimesh materials from SH3D data.
    Handles both PBR and simple material paths.
    """

    def __init__(self, archive: SH3DArchive):
        self._archive = archive
        self._texture_cache: Dict[str, Image.Image] = {}
        self._material_cache: Dict[str, PBRMaterial] = {}

    def load_texture_image(
        self, texture: SH3DTexture
    ) -> Optional[Image.Image]:
        """Load a texture image from the archive, with caching."""
        if not texture.image_path:
            return None

        if texture.image_path in self._texture_cache:
            return self._texture_cache[texture.image_path]

        try:
            data = self._archive.extract_texture_image(
                texture.image_path
            )
            img = Image.open(io.BytesIO(data))
            # Convert to RGBA for consistency
            img = img.convert("RGBA")
            self._texture_cache[texture.image_path] = img
            return img
        except Exception as e:
            logger.warning(
                f"Failed to load texture '{texture.image_path}': {e}"
            )
            return None

    def create_pbr_material(
        self,
        name: str,
        color: Optional[Tuple[int, int, int, int]] = None,
        texture: Optional[SH3DTexture] = None,
        shininess: float = 0.0,
        opacity: float = 1.0,
    ) -> PBRMaterial:
        """
        Create a GLTF-compatible PBR material.

        Parameters
        ----------
        name : str
            Material name.
        color : tuple, optional
            RGBA color (0-255).
        texture : SH3DTexture, optional
            Texture to apply as base color map.
        shininess : float
            SH3D shininess (0=matte, 1=mirror).
        opacity : float
            Overall opacity (0=transparent, 1=opaque).
        """
        cache_key = f"{name}_{color}_{texture.image_path if texture else ''}_{shininess}"
        if cache_key in self._material_cache:
            return self._material_cache[cache_key]

        base_color = color_to_basecolor(color) if color else np.array(
            [0.8, 0.8, 0.8, 1.0]
        )
        if opacity < 1.0:
            base_color[3] = opacity

        roughness = shininess_to_roughness(shininess)

        tex_image = None
        if texture is not None:
            tex_image = self.load_texture_image(texture)

        mat = PBRMaterial(
            name=name,
            baseColorFactor=base_color,
            baseColorTexture=tex_image,
            roughnessFactor=roughness,
            metallicFactor=0.0,  # architectural surfaces are non-metallic
            alphaMode="BLEND" if opacity < 1.0 else "OPAQUE",
            doubleSided=False,
        )

        self._material_cache[cache_key] = mat
        return mat

    def apply_furniture_materials(
        self,
        mesh: trimesh.Trimesh,
        piece: Furniture,
    ) -> trimesh.Trimesh:
        """
        Apply SH3D material overrides to a loaded furniture mesh.

        Priority:
        1. Per-material overrides (piece.material_overrides)
        2. Global color override (piece.color)
        3. Global texture override (piece.texture)
        4. Keep original model materials
        """
        if not isinstance(mesh, trimesh.Trimesh):
            # For Scenes, apply to each geometry
            if isinstance(mesh, trimesh.Scene):
                for geom_name, geom in mesh.geometry.items():
                    self._apply_to_single_mesh(geom, piece, geom_name)
            return mesh

        self._apply_to_single_mesh(mesh, piece, mesh.metadata.get("name", ""))
        return mesh

    def _apply_to_single_mesh(
        self,
        mesh: trimesh.Trimesh,
        piece: Furniture,
        mesh_name: str,
    ) -> None:
        """Apply material overrides to a single trimesh."""

        # Check for per-material override
        override = None
        for mat_override in piece.material_overrides:
            if (
                mat_override.name == mesh_name
                or mat_override.key == mesh_name
            ):
                override = mat_override
                break

        if override is not None:
            if not override.visible:
                # Hide this sub-mesh by making it fully transparent
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=mesh,
                    face_colors=np.zeros((len(mesh.faces), 4)),
                )
                return

            mat = self.create_pbr_material(
                name=f"{piece.name}_{override.name}",
                color=override.color,
                texture=override.texture,
                shininess=override.shininess,
                opacity=override.opacity,
            )
            try:
                mesh.visual = trimesh.visual.TextureVisuals(
                    material=mat
                )
            except Exception:
                pass
            return

        # Global color override
        if piece.color is not None:
            mat = self.create_pbr_material(
                name=f"{piece.name}_color",
                color=piece.color,
                shininess=piece.shininess,
            )
            try:
                mesh.visual = trimesh.visual.TextureVisuals(
                    material=mat
                )
            except Exception:
                # Fallback: just set vertex colors
                color_float = rgba_to_float(piece.color)
                if color_float:
                    mesh.visual = trimesh.visual.ColorVisuals(
                        mesh=mesh,
                        vertex_colors=np.tile(
                            color_float, (len(mesh.vertices), 1)
                        ),
                    )
            return

        # Global texture override
        if piece.texture is not None:
            mat = self.create_pbr_material(
                name=f"{piece.name}_texture",
                texture=piece.texture,
                shininess=piece.shininess,
            )
            try:
                mesh.visual = trimesh.visual.TextureVisuals(
                    material=mat
                )
            except Exception:
                pass

    def create_wall_materials(
        self, wall: Wall
    ) -> Tuple[PBRMaterial, PBRMaterial, PBRMaterial]:
        """
        Create materials for a wall's left side, right side,
        and top surface.
        """
        left_mat = self.create_pbr_material(
            name=f"wall_{wall.id}_left",
            color=wall.left_side_color,
            texture=wall.left_side_texture,
            shininess=wall.left_side_shininess,
        )
        right_mat = self.create_pbr_material(
            name=f"wall_{wall.id}_right",
            color=wall.right_side_color,
            texture=wall.right_side_texture,
            shininess=wall.right_side_shininess,
        )
        top_mat = self.create_pbr_material(
            name=f"wall_{wall.id}_top",
            color=wall.top_color,
        )
        return left_mat, right_mat, top_mat

    def create_floor_material(self, room: Room) -> PBRMaterial:
        """Create a material for a room's floor."""
        return self.create_pbr_material(
            name=f"room_{room.id}_floor",
            color=room.floor_color,
            texture=room.floor_texture,
            shininess=room.floor_shininess,
        )

    def create_ceiling_material(self, room: Room) -> PBRMaterial:
        """Create a material for a room's ceiling."""
        return self.create_pbr_material(
            name=f"room_{room.id}_ceiling",
            color=room.ceiling_color,
            texture=room.ceiling_texture,
            shininess=room.ceiling_shininess,
        )

    def create_glass_material(self, name: str = "glass") -> PBRMaterial:
        """
        Create a PBR glass material: clear, slightly blue-tinted,
        high roughness=0 (mirror-like), with BLEND transparency.
        """
        cache_key = f"__glass_{name}"
        if cache_key in self._material_cache:
            return self._material_cache[cache_key]

        mat = PBRMaterial(
            name=name,
            baseColorFactor=np.array([0.75, 0.88, 0.95, 0.18], dtype=np.float64),
            roughnessFactor=0.05,
            metallicFactor=0.0,
            alphaMode="BLEND",
            doubleSided=True,
        )
        self._material_cache[cache_key] = mat
        return mat

    def create_door_frame_material(self, name: str = "door_frame") -> PBRMaterial:
        """
        Create a neutral opaque material for door frames when
        the original model material cannot be preserved.
        """
        return self.create_pbr_material(
            name=name,
            color=(180, 160, 130, 255),  # warm wood-ish tone
            shininess=0.1,
        )