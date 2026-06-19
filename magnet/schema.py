from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator

class LinkSchema(BaseModel):
    title: str
    url: str
    type: str

class SubmitterSchema(BaseModel):
    name: str
    email: str

# TODO: this can be validated with a syntax check
class ClaimSchema(BaseModel):
    python: str

# TODO: not actually sure what the schema for "symbols" should look like.
class SymbolSchema(BaseModel):
    type: Optional[str] = None # TODO: should type be required?
    value: Optional[Any] = None
    sweep: Optional[list] = None
    depends_on: list[str] = Field(default_factory=list) # TODO: modify "depends_on" to reference an actual symbol
    python: Optional[str] = None

    @model_validator(mode='after')
    def has_resolution(self) -> 'SymbolSchema':
        if self.value is None and self.sweep is None and self.python is None:
            raise ValueError(
                "symbol must define at least one of: 'value', 'sweep', or 'python'"
            )
        return self

# TODO: If type == fraction, check that parameters[threshold] is defined and a float
class ClaimAggregationStrategySchema(BaseModel): 
    type: str
    model_config = {'extra': 'allow'} # without this, seems like the extra fields disappear

class EvaluationCardSchema(BaseModel):
    """
    Schema for an Evaluation Card YAML.

    Required fields: ``title``, ``description``, ``claim``.
    All other fields are optional to allow cards at different stages of
    completeness.

    Example:
        >>> import yaml
        >>> from magnet.schema import EvaluationCardSchema
        >>> raw = yaml.safe_load('''
        ...   title: "Arithmetic"
        ...   description: "Addition is commutative"
        ...   version: 1.0
        ...   organizations:
        ...     - Kitware
        ...   submitter:
        ...     name: Kitware TA2 Team
        ...     email: aiq-ta2@kitware.com
        ...   tags:
        ...     - example
        ...   links:
        ...     - title: "MAGNET"
        ...       url: "https://github.com/AIQ-Kitware/aiq-magnet"
        ...       type: "software"

        ...   claim:
        ...     python: "assert 1 + 2 == 2 + 1"
        ...   symbols:
        ...     x:
        ...       type: int
        ...       value: 1
        ... ''')
        >>> card = EvaluationCardSchema.model_validate(raw)
        >>> card.title
        'Arithmetic'
    """

    # --- Required ---
    title: str
    description: str
    claim: ClaimSchema
    version: str = Field(coerce_numbers_to_str=True)
    organizations: list[str]
    submitter: SubmitterSchema
    tags: list[str]
    links: list[LinkSchema]

    # --- Recommended ---
    category: Optional[str] = None

    # --- Evaluation configuration ---
    claim_aggregation_strategy: Optional[ClaimAggregationStrategySchema] = None
    symbols: Optional[dict[str, SymbolSchema]] = None

    # --- Backend (at most one) ---
    kwdagger: Optional[dict[str, Any]] = None
    pipeline: Optional[dict[str, Any]] = None

    @model_validator(mode='after')
    def exclusive_backends(self) -> 'EvaluationCardSchema':
        if self.kwdagger is not None and self.pipeline is not None:
            raise ValueError(
                "at most one of 'kwdagger' and 'pipeline' may be specified"
            )
        if self.kwdagger is None and self.pipeline is None and self.symbols is None:
            raise ValueError(
                "if 'pipeline'/'kwdagger' undefined, 'symbols' must be defined"
            )
        return self