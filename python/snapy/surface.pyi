"""
Stub file for snapy.surface module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch


# Surface
class SurfaceOptions:
    """
    Surface configuration options.

    This class manages surface parameters.
    """

    def __init__(self) -> None:
        """Initialize Surface with default values."""
        ...

    def __repr__(self) -> str: ...

    @staticmethod
    def from_yaml(filename: str, verbose: bool = False) -> "SurfaceOptions":
        """
        Load SurfaceOptions from a YAML file.

        Args:
            filename: Path to YAML file
            verbose: Enable verbose output

        Returns:
            SurfaceOptions loaded from file
        """
        ...

    @overload
    def diameters(self):  # Diameters
        """Get diameter sizes."""
        ...

    @overload
    def diameters(self, value) -> List[float]:  # Diameters
        """Set diameter sizes."""
        ...

class Surface:
    """
    Surface implementation.

    This module represents a surface dust supply.
    """

    @overload
    def __init__(self, options: SurfaceOptions, meshblock = None) -> None:
        """
        Construct a Surface module.

        Args:
            options: Surface configuration options
            meshblock: Parent MeshBlock object
        """
        ...

    def __repr__(self) -> str: ...

    options: SurfaceOptions

    def nbins(self) -> int:
        """
        Number of diameter bins (= num diameters - 1).
        """
        ...

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...
