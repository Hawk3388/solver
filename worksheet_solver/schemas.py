"""Structured data exchanged with the vision-language model."""

from typing import List

from pydantic import BaseModel, Field


class Pair(BaseModel):
    key: int = Field(strict=True)
    value: str = Field(strict=True)


class get_solution(BaseModel):
    solutions: List[Pair]
