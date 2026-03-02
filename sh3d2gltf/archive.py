# ============================================================
# archive.py - .sh3d ZIP extraction and XML parsing
# ============================================================

"""
Opens a .sh3d file (which is a ZIP archive), parses Home.xml,
extracts embedded models and textures, and returns a fully
populated Home dataclass.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, BinaryIO, Union
from xml.etree import ElementTree as ET

from .models import (
    Home, Level, Furniture, Wall, Room, Camera,
    Compass, Environment,
)


class SH3DArchive:
    """
    Represents an opened .sh3d file.
    Provides access to the parsed Home and raw ZIP contents.
    """

    def __init__(
        self,
        path: Union[str, Path, BinaryIO],
        temp_dir: Optional[str] = None,
    ):
        if isinstance(path, (str, Path)):
            self._zip = zipfile.ZipFile(str(path), "r")
            self._source_path = Path(path)
        else:
            self._zip = zipfile.ZipFile(path, "r")
            self._source_path = None

        self._temp_dir = temp_dir or tempfile.mkdtemp(
            prefix="sh3d2gltf_"
        )
        self._home: Optional[Home] = None
        self._asset_cache: Dict[str, bytes] = {}

    # ── Context manager ──────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._zip.close()

    # ── Public API ───────────────────────────────────────────

    @property
    def home(self) -> Home:
        """Parse Home.xml lazily and return the Home object."""
        if self._home is None:
            self._home = self._parse_home_xml()
        return self._home

    def list_entries(self):
        """List all file entries in the .sh3d ZIP."""
        return self._zip.namelist()

    def read_entry(self, name: str) -> bytes:
        """Read a raw file from the archive, with caching."""
        if name not in self._asset_cache:
            self._asset_cache[name] = self._zip.read(name)
        return self._asset_cache[name]

    def extract_entry(self, name: str, dest_dir: Optional[str] = None) -> str:
        """Extract one file to disk and return the path."""
        dest = dest_dir or self._temp_dir
        target = os.path.join(dest, os.path.basename(name))
        data = self.read_entry(name)
        with open(target, "wb") as f:
            f.write(data)
        return target

    def extract_model(self, model_path: str) -> str:
        """
        Extract a 3D model file (OBJ, DAE, 3DS, etc.)
        and any sibling files (MTL, textures) that share the
        same directory prefix in the ZIP.
        Returns path to the main model file.
        """
        if not model_path:
            raise ValueError("Empty model path")

        # Determine the "directory" prefix inside the ZIP
        # SH3D uses Content-digest paths like "1234567890/model.obj"
        prefix = ""
        if "/" in model_path:
            prefix = model_path.rsplit("/", 1)[0] + "/"

        # Extract all files with the same prefix
        model_dir = os.path.join(self._temp_dir, prefix.replace("/", "_"))
        os.makedirs(model_dir, exist_ok=True)

        extracted_main = None
        for entry in self._zip.namelist():
            if entry.startswith(prefix) or entry == model_path:
                basename = os.path.basename(entry)
                if not basename:
                    continue
                target = os.path.join(model_dir, basename)
                data = self.read_entry(entry)
                with open(target, "wb") as f:
                    f.write(data)
                if entry == model_path:
                    extracted_main = target

        if extracted_main is None:
            # Try exact match
            extracted_main = self.extract_entry(model_path, model_dir)

        return extracted_main

    def extract_texture_image(self, image_path: str) -> bytes:
        """Read texture image bytes from the archive."""
        return self.read_entry(image_path)

    def get_texture_path(self, image_path: str) -> str:
        """Extract a texture to disk and return the file path."""
        return self.extract_entry(image_path)

    # ── Internal parsing ─────────────────────────────────────

    def _parse_home_xml(self) -> Home:
        """
        Parse the Home.xml file inside the archive.
        Handles both <home> root and various SH3D versions.
        """
        # Find the XML file
        xml_name = None
        for name in self._zip.namelist():
            lower = name.lower()
            if lower == "home.xml" or lower.endswith("/home.xml"):
                xml_name = name
                break

        if xml_name is None:
            raise ValueError(
                "No Home.xml found in the .sh3d archive. "
                f"Contents: {self._zip.namelist()[:20]}"
            )

        xml_data = self._zip.read(xml_name)
        root = ET.fromstring(xml_data)

        home = Home()
        home.version = root.get("version", "")
        home.name = root.get("name", "")
        home.unit = root.get("unit", "CENTIMETER")

        # ── Levels ──
        for el in root.findall(".//level"):
            home.levels.append(Level.from_element(el))

        # Sort levels by elevation
        home.levels.sort(key=lambda lv: lv.elevation)

        # ── Furniture (all types) ──
        furniture_tags = [
            "pieceOfFurniture", "doorOrWindow",
            "light", "furnitureGroup",
        ]
        for tag in furniture_tags:
            for el in root.findall(f".//{tag}"):
                # Avoid double-counting children of groups that
                # were already parsed recursively
                parent = _find_parent(root, el)
                if parent is not None and parent.tag == "furnitureGroup":
                    continue
                home.furniture.append(Furniture.from_element(el))

        # ── Walls ──
        for el in root.findall(".//wall"):
            home.walls.append(Wall.from_element(el))

        # ── Rooms ──
        for el in root.findall(".//room"):
            home.rooms.append(Room.from_element(el))

        # ── Cameras ──
        for tag in ("observerCamera", "camera", "storedCamera"):
            for el in root.findall(f".//{tag}"):
                home.cameras.append(Camera.from_element(el))

        # ── Compass ──
        compass_el = root.find(".//compass")
        if compass_el is not None:
            home.compass = Compass.from_element(compass_el)

        # ── Environment ──
        env_el = root.find(".//environment")
        if env_el is not None:
            home.environment = Environment.from_element(env_el)

        home.build_indices()
        return home


def _find_parent(root: ET.Element, target: ET.Element) -> Optional[ET.Element]:
    """
    Find the parent of `target` within the tree rooted at `root`.
    ElementTree doesn't natively support parent lookup so we do
    a simple DFS with parent tracking.
    """
    for parent in root.iter():
        for child in parent:
            if child is target:
                return parent
    return None