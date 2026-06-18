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

class ClaimAggregationStrategySchema(BaseModel): 
    type: str
    # TODO: check if this is the idiomatic way to let a field contain whatever
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

    # --- Recommended metadata ---
    category: Optional[str] = None
    version: Optional[str] = None
    organizations: Optional[list[str]] = None
    submitter: Optional[SubmitterSchema] = None
    tags: Optional[list[str]] = None
    links: Optional[list[LinkSchema]] = None

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
        return self