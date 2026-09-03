"""Typed JSON records shared by the SmolLM example commands."""

from typing import TypedDict


class QuestionItem(TypedDict):
    id: int
    prompt: str
    expected: str


class ItemsMetrics(TypedDict):
    n_items: int
    seed: int


class ItemsResult(TypedDict):
    metrics: ItemsMetrics


class ItemsPayload(TypedDict):
    result: ItemsResult
    items: list[QuestionItem]


class AnswerRecord(TypedDict):
    id: int
    expected: str
    answer: str
    normalized: str
    seconds: float
    error: str


class AskMetrics(TypedDict):
    endpoint: str
    served_model: str
    n_items: int
    n_answered: int
    answered_rate: float
    exact_rate: float
    mean_seconds: float


class AskResult(TypedDict):
    metrics: AskMetrics


class AnswersPayload(TypedDict):
    result: AskResult
    answers: list[AnswerRecord]


class CompareMetrics(TypedDict):
    n_endpoints: int
    endpoints: str
    n_items: int
    coverage: float
    agreement: float


class CompareResult(TypedDict):
    metrics: CompareMetrics


class ComparePayload(TypedDict):
    result: CompareResult
