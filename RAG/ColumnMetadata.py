from RAG.DataClassBase import DataClassBase
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ColumnMetadata(DataClassBase):
    """Example implementation for database column metadata"""
    table_name: str
    column_name: str
    column_type: str
    column_remarks: str

    def to_text(self) -> Dict[str, str]:
        return {
            'name': self.column_name,
            'type': self.column_type,
            'remarks': self.column_remarks,
            'context': f"{self.table_name} {self.column_name}"
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'table_name': self.table_name,
            'column_name': self.column_name,
            'column_type': self.column_type,
            'column_remarks': self.column_remarks
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ColumnMetadata':
        return cls(**data)
