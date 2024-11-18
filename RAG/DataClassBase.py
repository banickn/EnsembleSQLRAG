from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar

# Generic type for metadata
T = TypeVar('T')


class DataClassBase(ABC):
    """Abstract base class for metadata objects"""
    @abstractmethod
    def to_text(self) -> Dict[str, str]:
        """Convert metadata to text components for embedding"""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataClassBase':
        """Create metadata object from dictionary"""
        pass
