# ============================================================
# setup.py - Package installation
# ============================================================

from setuptools import setup, find_packages

setup(
    name="sh3d2gltf",
    version="1.0.0",
    description="Convert Sweet Home 3D (.sh3d) files to GLTF/GLB",
    long_description=open("README.md").read()
    if __import__("os").path.exists("README.md")
    else "",
    long_description_content_type="text/markdown",
    author="sh3d2gltf contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "trimesh[easy]>=3.15",
        "Pillow>=8.0",
        "shapely>=1.8",
    ],
    extras_require={
        "full": [
            "pygltflib>=1.15",
            "lxml>=4.6",
            "networkx>=2.5",
            "scipy>=1.7",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "sh3d2gltf=sh3d2gltf.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
)