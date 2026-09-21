"""Structured data exchanged with the vision-language model."""

from typing import List

from pydantic import BaseModel


class Pair(BaseModel):
    key: int
    value: str


class get_solution(BaseModel):
    solutions: List[Pair]
