# ============================================================
# converter.py - Top-level orchestrator
# ============================================================

"""
The main conversion function. This is what you call from
your code or from the CLI.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, BinaryIO

from .archive import SH3DArchive
from .scene_builder import SceneBuilder
from .gltf_export import export_gltf
from .utils import CM_TO_M, unit_to_meters, logger


@dataclass
class ConversionOptions:
    """Configuration for the SH3D → GLTF conversion."""

    # Output format
    embed_textures: bool = True
    output_format: str = "glb"      # "glb" or "gltf"

    # Content options
    include_furniture: bool = True
    include_walls: bool = True
    include_rooms: bool = True
    include_cameras: bool = True
    include_lights: bool = True
    include_invisible: bool = False

    # Quality options
    draco_compression: bool = False
    merge_duplicate_vertices: bool = True

    # Coordinate options
    unit_scale: Optional[float] = None  # auto-detect from file
    y_up: bool = True               # GLTF standard

    # Debugging
    log_level: str = "INFO"
    keep_temp_files: bool = False
    temp_dir: Optional[str] = None


def convert_sh3d_to_gltf(
    input_path: Union[str, Path, BinaryIO],
    output_path: Union[str, Path],
    options: Optional[ConversionOptions] = None,
) -> str:
    """
    Convert a Sweet Home 3D file to GLTF/GLB.

    Parameters
    ----------
    input_path : str, Path, or file-like
        Path to the .sh3d file or an open binary stream.
    output_path : str or Path
        Where to write the output .gltf or .glb file.
    options : ConversionOptions, optional
        Conversion settings. Uses defaults if not provided.

    Returns
    -------
    str
        Absolute path to the output file.

    Examples
    --------
    >>> from sh3d2gltf import convert_sh3d_to_gltf
    >>> convert_sh3d_to_gltf("my_house.sh3d", "my_house.glb")
    '/absolute/path/to/my_house.glb'

    >>> from sh3d2gltf import convert_sh3d_to_gltf, ConversionOptions
    >>> opts = ConversionOptions(
    ...     draco_compression=True,
    ...     include_lights=False,
    ... )
    >>> convert_sh3d_to_gltf("house.sh3d", "house.gltf", opts)
    """
    if options is None:
        options = ConversionOptions()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, options.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    output_path = Path(output_path)

    # Auto-detect output format from extension
    if output_path.suffix.lower() == ".gltf":
        options.output_format = "gltf"
    elif output_path.suffix.lower() == ".glb":
        options.output_format = "glb"

    # Create temp directory
    temp_dir = options.temp_dir or tempfile.mkdtemp(prefix="sh3d2gltf_")

    logger.info(f"Converting: {input_path} → {output_path}")

    try:
        # ── Step 1: Open and parse the archive ───────────────
        with SH3DArchive(input_path, temp_dir=temp_dir) as archive:
            home = archive.home

            logger.info(
                f"Parsed home: version={home.version}, "
                f"unit={home.unit}, "
                f"{len(home.levels)} levels, "
                f"{len(home.furniture)} furniture, "
                f"{len(home.walls)} walls, "
                f"{len(home.rooms)} rooms"
            )

            # Determine unit scale
            if options.unit_scale is not None:
                unit_scale = options.unit_scale
            else:
                unit_scale = unit_to_meters(home.unit)

            # ── Step 2: Build the scene ──────────────────────
            builder = SceneBuilder(
                archive=archive,
                unit_scale=unit_scale,
                include_cameras=options.include_cameras,
                include_lights=options.include_lights,
                include_invisible=options.include_invisible,
            )

            scene = builder.build()

            # ── Step 3: Export to GLTF/GLB ───────────────────
            result = export_gltf(
                scene=scene,
                output_path=str(output_path),
                embed_textures=options.embed_textures,
                add_lights=options.include_lights,
                add_cameras=options.include_cameras,
                draco_compression=options.draco_compression,
            )

            logger.info(f"Conversion complete: {result}")
            return result

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        raise

    finally:
        # Cleanup temp files
        if not options.keep_temp_files:
            import shutil
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass