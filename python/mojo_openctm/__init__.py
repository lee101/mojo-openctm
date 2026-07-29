"""OpenCTM MG1/MG2 mesh compression accelerated by Mojo."""

from .codec import (
    AttributeMap,
    CTMError,
    Mesh,
    UVMap,
    dumps,
    load,
    loads,
    save,
)

__version__ = "0.1.0"
__all__ = [
    "AttributeMap",
    "CTMError",
    "Mesh",
    "UVMap",
    "dumps",
    "load",
    "loads",
    "save",
]
