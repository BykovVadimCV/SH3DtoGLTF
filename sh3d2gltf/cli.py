# ============================================================
# cli.py - Command-line interface
# ============================================================

"""
CLI entry point for sh3d2gltf.

Usage:
    python -m sh3d2gltf input.sh3d output.glb [options]
    sh3d2gltf input.sh3d output.glb [options]
"""

from __future__ import annotations

import argparse
import sys
import time

from .converter import convert_sh3d_to_gltf, ConversionOptions


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="sh3d2gltf",
        description=(
            "Convert Sweet Home 3D (.sh3d) files to GLTF/GLB format. "
            "Preserves furniture placement, materials, textures, "
            "lights, cameras, and multi-level hierarchies."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  sh3d2gltf house.sh3d house.glb\n"
            "  sh3d2gltf house.sh3d house.gltf --no-lights --draco\n"
            "  sh3d2gltf house.sh3d house.glb --embed --verbose\n"
        ),
    )

    parser.add_argument(
        "input",
        help="Input .sh3d file path",
    )
    parser.add_argument(
        "output",
        help="Output .gltf or .glb file path",
    )

    # Content options
    content = parser.add_argument_group("Content options")
    content.add_argument(
        "--no-furniture",
        action="store_true",
        help="Exclude furniture from output",
    )
    content.add_argument(
        "--no-walls",
        action="store_true",
        help="Exclude walls from output",
    )
    content.add_argument(
        "--no-rooms",
        action="store_true",
        help="Exclude rooms (floors/ceilings) from output",
    )
    content.add_argument(
        "--no-lights",
        action="store_true",
        help="Exclude light sources",
    )
    content.add_argument(
        "--no-cameras",
        action="store_true",
        help="Exclude cameras",
    )
    content.add_argument(
        "--include-invisible",
        action="store_true",
        help="Include items marked as invisible in SH3D",
    )

    # Format options
    fmt = parser.add_argument_group("Format options")
    fmt.add_argument(
        "--embed",
        action="store_true",
        default=True,
        help="Embed textures in GLB (default: True)",
    )
    fmt.add_argument(
        "--no-embed",
        action="store_true",
        help="Don't embed textures (external references)",
    )
    fmt.add_argument(
        "--draco",
        action="store_true",
        help="Enable Draco mesh compression (requires pygltflib)",
    )

    # Debug options
    debug = parser.add_argument_group("Debug options")
    debug.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    debug.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )
    debug.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary extracted files",
    )
    debug.add_argument(
        "--temp-dir",
        help="Custom temporary directory path",
    )

    args = parser.parse_args(argv)

    # Build options
    options = ConversionOptions(
        include_furniture=not args.no_furniture,
        include_walls=not args.no_walls,
        include_rooms=not args.no_rooms,
        include_lights=not args.no_lights,
        include_cameras=not args.no_cameras,
        include_invisible=args.include_invisible,
        embed_textures=not args.no_embed,
        draco_compression=args.draco,
        keep_temp_files=args.keep_temp,
        temp_dir=args.temp_dir,
    )

    if args.verbose:
        options.log_level = "DEBUG"
    elif args.quiet:
        options.log_level = "ERROR"

    # Run conversion
    start = time.time()
    try:
        result = convert_sh3d_to_gltf(
            args.input, args.output, options
        )
        elapsed = time.time() - start

        if not args.quiet:
            print(f"\n✓ Conversion complete in {elapsed:.2f}s")
            print(f"  Output: {result}")

    except FileNotFoundError as e:
        print(f"✗ File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Invalid input: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"✗ Conversion failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()