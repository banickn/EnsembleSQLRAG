from RAG.DataClassBase import DataClassBase
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class QueryData(DataClassBase):
    """Example implementation for database queries"""
    user: str
    query: str
    tables: list[str]

    def to_text(self) -> Dict[str, str]:
        return {
            'user': self.user,
            'query': self.query,
            # Convert list of tables to space-separated string
            'tables': ' '.join(self.tables)
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'user': self.user,
            'query': self.query,
            'tables': self.tables
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryData':
        return cls(**data)
