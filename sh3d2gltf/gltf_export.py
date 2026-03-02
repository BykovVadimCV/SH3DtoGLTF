# ============================================================
# gltf_export.py - GLTF/GLB export with extensions
# ============================================================

"""
Handles the final export step. Uses trimesh's built-in GLTF
exporter for the base file, then optionally enhances it with:
- KHR_lights_punctual for light sources
- KHR_draco_mesh_compression for smaller files
- Embedded textures (GLB) or external references (GLTF)
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import trimesh

from .utils import logger


def _inline_external_refs(
    gltf_dict: dict,
    side_files: Dict[str, bytes],
) -> dict:
    """
    Walk a parsed GLTF JSON dict and replace every external URI
    (buffer or image) that appears in *side_files* with a base64
    data URI.  This makes the .gltf file fully self-contained — no
    companion .bin or image files are needed.

    Parameters
    ----------
    gltf_dict : dict
        Parsed GLTF JSON (mutated in-place and returned).
    side_files : dict
        Mapping of bare filename → raw bytes as returned by trimesh
        when it exports to the ``"gltf"`` file type.

    Returns
    -------
    dict
        The same *gltf_dict* with all resolvable URIs inlined.
    """
    def _to_data_uri(filename: str, raw: bytes) -> str:
        # Choose the right MIME type from the file extension.
        ext = Path(filename).suffix.lower()
        mime_map = {
            ".bin":  "application/octet-stream",
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".ktx2": "image/ktx2",
        }
        mime = mime_map.get(ext, "application/octet-stream")
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    # Inline geometry buffers
    for buf in gltf_dict.get("buffers", []):
        uri = buf.get("uri", "")
        if uri and not uri.startswith("data:") and uri in side_files:
            original_size = len(side_files[uri])
            buf["uri"] = _to_data_uri(uri, side_files[uri])
            logger.debug(
                f"Inlined buffer '{uri}' "
                f"({original_size:,} bytes → data URI)"
            )

    # Inline image sources
    for image in gltf_dict.get("images", []):
        uri = image.get("uri", "")
        if uri and not uri.startswith("data:") and uri in side_files:
            original_size = len(side_files[uri])
            image["uri"] = _to_data_uri(uri, side_files[uri])
            logger.debug(
                f"Inlined image '{uri}' "
                f"({original_size:,} bytes → data URI)"
            )

    return gltf_dict


# GLTF bufferView target constants
_ARRAY_BUFFER = 34962          # vertex attribute data
_ELEMENT_ARRAY_BUFFER = 34963  # index data


def _fix_buffer_view_targets(gltf_dict: dict) -> dict:
    """
    Stamp the correct ``target`` field on every bufferView that backs
    vertex attribute or index data.

    trimesh leaves ``target`` unset, which is technically valid JSON but
    triggers ``BUFFER_VIEW_TARGET_MISSING`` warnings in every conformance
    checker (Khronos validator, gltf.report, etc.) and prevents some GPU
    drivers from making the optimal memory allocation hint.

    The algorithm:
    1. Walk every mesh → primitive.
    2. For each attribute accessor  → mark its bufferView as ARRAY_BUFFER.
    3. For the indices accessor      → mark its bufferView as ELEMENT_ARRAY_BUFFER.

    A bufferView that is used by *both* attribute and index data in
    different primitives (degenerate but legal) is left without a target
    since neither value would be correct for the whole view — in practice
    this never happens with trimesh output.

    Parameters
    ----------
    gltf_dict : dict
        Parsed GLTF JSON (mutated in-place and returned).

    Returns
    -------
    dict
        The same dict with bufferView targets filled in.
    """
    accessors   = gltf_dict.get("accessors", [])
    buffer_views = gltf_dict.get("bufferViews", [])

    # Collect intended targets for every bufferView index.
    # If conflicting assignments arise, fall back to no target.
    intended: dict[int, int] = {}
    conflicts: set[int] = set()

    def _assign(accessor_index: int, target: int) -> None:
        if accessor_index is None or accessor_index >= len(accessors):
            return
        acc = accessors[accessor_index]
        bv_idx = acc.get("bufferView")
        if bv_idx is None or bv_idx >= len(buffer_views):
            return
        if bv_idx in conflicts:
            return
        if bv_idx in intended and intended[bv_idx] != target:
            conflicts.add(bv_idx)
            del intended[bv_idx]
            return
        intended[bv_idx] = target

    for mesh in gltf_dict.get("meshes", []):
        for prim in mesh.get("primitives", []):
            # Vertex attributes → ARRAY_BUFFER
            for accessor_idx in prim.get("attributes", {}).values():
                _assign(accessor_idx, _ARRAY_BUFFER)
            # Index data → ELEMENT_ARRAY_BUFFER
            if "indices" in prim:
                _assign(prim["indices"], _ELEMENT_ARRAY_BUFFER)

    # Also handle morph target attributes
    for mesh in gltf_dict.get("meshes", []):
        for prim in mesh.get("primitives", []):
            for target_dict in prim.get("targets", []):
                for accessor_idx in target_dict.values():
                    _assign(accessor_idx, _ARRAY_BUFFER)

    # Apply
    stamped = 0
    for bv_idx, target in intended.items():
        bv = buffer_views[bv_idx]
        if "target" not in bv:
            bv["target"] = target
            stamped += 1

    if stamped:
        logger.debug(
            f"Stamped bufferView.target on {stamped} buffer view(s) "
            f"({len(conflicts)} conflict(s) left unset)."
        )

    return gltf_dict


def _patch_glb_buffer_view_targets(glb_bytes: bytes) -> bytes:
    """
    Parse a GLB binary, fix bufferView targets in the JSON chunk via
    ``_fix_buffer_view_targets``, and reassemble the binary.

    GLB binary layout (little-endian):
        Bytes 0–11  : header  (magic 0x46546C67, version, total length)
        Bytes 12–…  : chunks  (length uint32, type uint32, data)
                      Chunk type 0x4E4F534A = JSON
                      Chunk type 0x004E4942 = BIN

    We only touch the JSON chunk; the BIN chunk is passed through unchanged.
    """
    MAGIC       = 0x46546C67
    JSON_TYPE   = 0x4E4F534A
    BIN_TYPE    = 0x004E4942

    if len(glb_bytes) < 12:
        raise ValueError("GLB too short to be valid")

    magic, version, _total_len = struct.unpack_from("<III", glb_bytes, 0)
    if magic != MAGIC:
        raise ValueError(f"Not a GLB file (magic={magic:#010x})")

    # Parse chunks
    offset = 12
    chunks = []  # list of (chunk_type, chunk_data_bytes)
    while offset < len(glb_bytes):
        if offset + 8 > len(glb_bytes):
            break
        chunk_len, chunk_type = struct.unpack_from("<II", glb_bytes, offset)
        offset += 8
        chunk_data = glb_bytes[offset: offset + chunk_len]
        offset += chunk_len
        chunks.append((chunk_type, bytearray(chunk_data)))

    # Fix JSON chunk
    patched_chunks = []
    for chunk_type, chunk_data in chunks:
        if chunk_type == JSON_TYPE:
            # JSON chunk is padded to 4-byte alignment with spaces (0x20)
            json_str = chunk_data.rstrip(b" \x00").decode("utf-8")
            gltf_dict = json.loads(json_str)
            gltf_dict = _fix_buffer_view_targets(gltf_dict)
            # Re-encode and re-pad to 4-byte boundary with spaces
            new_json = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
            pad = (4 - len(new_json) % 4) % 4
            new_json += b" " * pad
            patched_chunks.append((chunk_type, new_json))
        else:
            patched_chunks.append((chunk_type, bytes(chunk_data)))

    # Reassemble
    body = b""
    for chunk_type, chunk_data in patched_chunks:
        body += struct.pack("<II", len(chunk_data), chunk_type)
        body += chunk_data

    total_len = 12 + len(body)
    header = struct.pack("<III", MAGIC, version, total_len)
    return header + body


def export_gltf(
    scene: trimesh.Scene,
    output_path: str,
    embed_textures: bool = True,
    add_lights: bool = True,
    add_cameras: bool = True,
    draco_compression: bool = False,
    pretty_print: bool = False,
) -> str:
    """
    Export a trimesh Scene to a single, self-contained GLTF or GLB file.

    For ``.glb`` output the standard binary container is used — JSON
    header, binary buffer chunk, and (when *embed_textures* is True)
    all texture data are packed into one file by trimesh.

    For ``.gltf`` output trimesh normally produces a JSON file **plus**
    separate ``.bin`` / image files.  This function post-processes that
    result and re-encodes every external reference as a base64 data URI
    (``data:application/octet-stream;base64,…`` for buffers,
    ``data:image/png;base64,…`` for PNG images, etc.) so that the final
    ``.gltf`` is completely self-contained.  No companion files are
    written to disk.

    Note: data-URI embedding increases file size by roughly 33 % compared
    to a binary GLB.  If file size is a concern, use ``.glb`` instead.

    Parameters
    ----------
    scene : trimesh.Scene
        The assembled scene.
    output_path : str
        Destination file path (``.gltf`` or ``.glb``).
    embed_textures : bool
        Passed through to trimesh for GLB packing; for GLTF output all
        textures are always inlined regardless of this flag.
    add_lights : bool
        If True, add ``KHR_lights_punctual`` from scene metadata.
    add_cameras : bool
        If True, add camera nodes from scene metadata.
    draco_compression : bool
        If True, flag ``KHR_draco_mesh_compression`` (requires pygltflib).
    pretty_print : bool
        For ``.gltf`` output, format the JSON with 2-space indentation.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    output = Path(output_path)

    # ── Default to .gltf if no recognised extension ──────────
    if output.suffix.lower() not in (".gltf", ".glb"):
        output = output.with_suffix(".gltf")

    output.parent.mkdir(parents=True, exist_ok=True)

    is_glb = output.suffix.lower() == ".glb"
    file_type = "glb" if is_glb else "gltf"

    # ── Export via trimesh ───────────────────────────────────
    try:
        exported = scene.export(file_type=file_type)
    except Exception as e:
        logger.error(f"trimesh export failed: {e}")
        raise

    # ── GLB path: trimesh returns raw bytes, fix targets, write ──
    if is_glb:
        if not isinstance(exported, bytes):
            # Shouldn't happen, but handle gracefully
            exported = json.dumps(exported).encode("utf-8")
        try:
            exported = _patch_glb_buffer_view_targets(exported)
        except Exception as e:
            logger.warning(f"Could not patch GLB bufferView targets: {e}")
        with open(output, "wb") as f:
            f.write(exported)
        logger.info(f"Exported GLB to {output} ({len(exported):,} bytes)")

    # ── GLTF path: inline all side-files as data URIs ────────
    else:
        # trimesh returns either:
        #   bytes  — already a complete JSON blob (rare)
        #   str    — JSON string, no side files
        #   dict   — {"model.gltf": <json bytes/str>, "buffer0.bin": <bytes>, …}
        if isinstance(exported, (bytes, str)):
            # No side files — write as-is (already self-contained)
            gltf_text = (
                exported.decode("utf-8")
                if isinstance(exported, bytes)
                else exported
            )
            if pretty_print:
                try:
                    gltf_text = json.dumps(
                        json.loads(gltf_text), indent=2
                    )
                except (json.JSONDecodeError, TypeError):
                    pass
            with open(output, "w", encoding="utf-8") as f:
                f.write(gltf_text)
            logger.info(f"Exported GLTF to {output}")

        elif isinstance(exported, dict):
            # Locate the main JSON entry — trimesh uses either the
            # output filename or the generic key "model.gltf".
            main_key = None
            for candidate in (output.name, "model.gltf"):
                if candidate in exported:
                    main_key = candidate
                    break

            if main_key is None:
                # Last resort: pick the first .gltf-like entry
                for k in exported:
                    if k.endswith(".gltf") or k.endswith(".json"):
                        main_key = k
                        break

            if main_key is None:
                # No recognisable JSON entry — dump the whole dict
                logger.warning(
                    "Could not identify GLTF JSON key in trimesh "
                    "export dict; writing raw JSON."
                )
                with open(output, "w", encoding="utf-8") as f:
                    json.dump(
                        exported, f,
                        indent=2 if pretty_print else None
                    )
            else:
                raw_json = exported[main_key]
                if isinstance(raw_json, bytes):
                    raw_json = raw_json.decode("utf-8")

                gltf_dict = json.loads(raw_json)

                # Collect all side-file bytes (skip the main JSON entry)
                side_files: Dict[str, bytes] = {}
                for filename, data in exported.items():
                    if filename == main_key:
                        continue
                    if isinstance(data, str):
                        data = data.encode("utf-8")
                    side_files[filename] = data

                if side_files:
                    logger.info(
                        f"Inlining {len(side_files)} external file(s) "
                        f"as data URIs: {list(side_files.keys())}"
                    )
                    gltf_dict = _inline_external_refs(gltf_dict, side_files)
                else:
                    logger.debug("No side files to inline.")

                gltf_dict = _fix_buffer_view_targets(gltf_dict)

                indent = 2 if pretty_print else None
                gltf_text = json.dumps(gltf_dict, indent=indent)

                with open(output, "w", encoding="utf-8") as f:
                    f.write(gltf_text)

                logger.info(
                    f"Exported self-contained GLTF to {output} "
                    f"({len(gltf_text):,} chars)"
                )
        else:
            logger.warning(
                f"Unexpected trimesh export type {type(exported)}; "
                f"attempting string conversion."
            )
            with open(output, "w", encoding="utf-8") as f:
                f.write(str(exported))

    # ── Post-process with pygltflib if available ─────────────
    if add_lights or add_cameras or draco_compression:
        try:
            _postprocess_gltf(
                output_path=str(output),
                scene_metadata=scene.metadata,
                add_lights=add_lights,
                add_cameras=add_cameras,
                draco_compression=draco_compression,
            )
        except ImportError:
            logger.info(
                "pygltflib not available — skipping post-processing "
                "(lights, cameras, Draco). Install with: "
                "pip install pygltflib"
            )
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")

    return str(output.resolve())


def _postprocess_gltf(
    output_path: str,
    scene_metadata: Dict[str, Any],
    add_lights: bool,
    add_cameras: bool,
    draco_compression: bool,
) -> None:
    """
    Use pygltflib to add extensions and features that
    trimesh doesn't natively support.
    """
    import pygltflib

    gltf = pygltflib.GLTF2().load(output_path)

    modified = False

    # ── Add KHR_lights_punctual ──────────────────────────────
    if add_lights and "lights" in scene_metadata:
        lights_data = scene_metadata["lights"]
        if lights_data:
            _add_lights_extension(gltf, lights_data)
            modified = True

    # ── Add cameras ──────────────────────────────────────────
    if add_cameras and "cameras" in scene_metadata:
        cameras_data = scene_metadata["cameras"]
        if cameras_data:
            _add_cameras_to_gltf(gltf, cameras_data)
            modified = True

    # ── Draco compression ────────────────────────────────────
    if draco_compression:
        try:
            if not gltf.extensionsUsed:
                gltf.extensionsUsed = []
            if "KHR_draco_mesh_compression" not in gltf.extensionsUsed:
                gltf.extensionsUsed.append("KHR_draco_mesh_compression")
            logger.info(
                "Draco extension flagged. Use gltf-transform or "
                "similar tool to actually compress meshes."
            )
            modified = True
        except Exception as e:
            logger.warning(f"Draco setup failed: {e}")

    if modified:
        gltf.save(output_path)
        logger.info(f"Post-processed GLTF saved to {output_path}")


def _add_lights_extension(
    gltf,
    lights_data: List[Dict[str, Any]],
) -> None:
    """Add KHR_lights_punctual extension to the GLTF."""
    import pygltflib

    ext_name = "KHR_lights_punctual"

    if not gltf.extensionsUsed:
        gltf.extensionsUsed = []
    if ext_name not in gltf.extensionsUsed:
        gltf.extensionsUsed.append(ext_name)

    lights = []
    for light in lights_data:
        lights.append({
            "type": light.get("type", "point"),
            "color": light.get("color", [1.0, 1.0, 1.0]),
            "intensity": light.get("intensity", 100.0),
            "name": light.get("name", "light"),
        })

    if not gltf.extensions:
        gltf.extensions = {}
    gltf.extensions[ext_name] = {"lights": lights}

    for i, light in enumerate(lights_data):
        pos = light.get("position", [0, 0, 0])
        node = pygltflib.Node(
            name=f"Light_{i}",
            translation=pos,
            extensions={ext_name: {"light": i}},
        )
        gltf.nodes.append(node)

        if gltf.scenes:
            gltf.scenes[0].nodes.append(len(gltf.nodes) - 1)

    logger.info(f"Added {len(lights)} lights via {ext_name}")


def _add_cameras_to_gltf(
    gltf,
    cameras_data: List[Dict[str, Any]],
) -> None:
    """Add camera nodes to the GLTF."""
    import pygltflib
    import math

    for i, cam_data in enumerate(cameras_data):
        fov = cam_data.get("fov", math.radians(63))

        camera = pygltflib.Camera(
            name=cam_data.get("name", f"Camera_{i}"),
            type="perspective",
            perspective=pygltflib.Perspective(
                aspectRatio=1.5,
                yfov=fov,
                znear=0.01,
                zfar=1000.0,
            ),
        )
        gltf.cameras.append(camera)

        pos = cam_data.get("position", [0, 1.7, 0])
        node = pygltflib.Node(
            name=cam_data.get("name", f"Camera_{i}"),
            camera=len(gltf.cameras) - 1,
            translation=pos,
        )
        gltf.nodes.append(node)

        if gltf.scenes:
            gltf.scenes[0].nodes.append(len(gltf.nodes) - 1)

    logger.info(f"Added {len(cameras_data)} cameras")