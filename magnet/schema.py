from enum import StrEnum
from typing import Any, Literal, Optional
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

class MetricObjective(StrEnum):
    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'

class MetricAggregationStrategySchema(BaseModel):
    type: Literal["mean", "max", "min", "custom"]
    parameters: dict[str, float] | None = None

class MetricSymbolSchema(BaseModel):
    objective: MetricObjective = MetricObjective.MAXIMIZE
    aggregation_strategy: MetricAggregationStrategySchema

class SymbolMetadataSchema(BaseModel):
    display: bool | None = None
    display_name: str | None = None
    define_metric: MetricSymbolSchema | None = None

class SymbolSchema(BaseModel):
    type: str | None = None
    value: Any | None = None
    sweep: list | None = None
    depends_on: list[str] = Field(default_factory=list) # TODO: modify "depends_on" to reference an actual symbol
    python: str | None = None # TODO: this can be validated with a syntax check
    metadata: SymbolMetadataSchema | None = None

    @model_validator(mode='after')
    def has_resolution(self) -> 'SymbolSchema':
        if self.value is None and self.sweep is None and self.python is None:
            if self.metadata is not None and self.metadata.define_metric is not None:
                # Handle metric definitions in kwdagger/pipeline cards 
                # (i.e. ignore test for symbols defined/calculated in user script)
                return self
            else:
                raise ValueError(
                    "symbol must define at least one of: 'value', 'sweep', or 'python'"
                )
        return self

class ClaimAggregationStrategyParameterSchema(BaseModel):
    threshold: float

# TODO: If type == fraction, check that parameters[threshold] is defined and a float
class ClaimAggregationStrategySchema(BaseModel):
    type: str
    parameters: ClaimAggregationStrategyParameterSchema | None = None
    model_config = {'extra': 'allow'} # without this, seems like the extra fields disappear

    @model_validator(mode='after')
    def confirm_threshold(self) -> 'ClaimAggregationStrategySchema':
        if self.type == "fraction" and (self.parameters is None or self.parameters.threshold is None):
            raise ValueError(
                "claim aggregation strategy of fraction requires threshold parameter"
            )
        return self

class TheoryFormalizationSchema(BaseModel):
    """Structured provenance for a formal theory source."""
    system: str
    repository: str | None = None
    revision: str | None = None
    model_config = {'extra': 'forbid'}

    @model_validator(mode='after')
    def pinned_repository(self) -> 'TheoryFormalizationSchema':
        if self.repository and not self.revision:
            raise ValueError('formalization.revision is required with repository')
        return self


class TheoryPremiseSchema(BaseModel):
    """A named premise, normally a proposition-valued formal binder."""
    id: str
    type: str | None = None
    statement: str | None = None
    model_config = {'extra': 'forbid'}


class TheoryEntrySchema(BaseModel):
    """A theoretical object empirical code can point at."""
    id: str
    kind: Literal[
        "theorem", "conjecture", "question", "definition"
    ]
    statement: str | None = None
    declaration: str | None = None
    formalization: TheoryFormalizationSchema | None = None
    source_path: str | None = None
    premises: list[TheoryPremiseSchema] = Field(default_factory=list)
    model_config = {'extra': 'forbid'}

    @model_validator(mode='after')
    def has_identity(self) -> 'TheoryEntrySchema':
        if not self.statement and not self.declaration:
            raise ValueError("theory entry needs a statement or declaration")
        if '::' in self.id:
            raise ValueError("theory entry id may not contain '::'")
        premise_ids = [premise.id for premise in self.premises]
        if len(premise_ids) != len(set(premise_ids)):
            raise ValueError('theory entry has duplicate premise ids')
        return self


class TheoryLinkSchema(BaseModel):
    """A card-level relationship to a whole theoretical statement."""
    relation: Literal["tests", "approximates", "motivates"]
    ref: str
    note: str | None = None
    model_config = {'extra': 'forbid'}

    @model_validator(mode='after')
    def targets_statement(self) -> 'TheoryLinkSchema':
        if '::' in self.ref:
            raise ValueError('card-level theory links cannot target premises')
        return self


class TheorySchema(BaseModel):
    """Static theory references and empirical source files for a card."""
    links: list[TheoryLinkSchema] = Field(default_factory=list)
    empirical_sources: list[str] = Field(default_factory=list)
    entries: list[TheoryEntrySchema] = Field(default_factory=list)
    indexes: list[str] = Field(default_factory=list)
    model_config = {'extra': 'forbid'}


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
    category: str | None = None

    # --- Evaluation configuration ---
    claim_aggregation_strategy: ClaimAggregationStrategySchema | None = None
    symbols: dict[str, SymbolSchema] | None = None

    # --- how the code relates to a theoretical object (optional) ---
    theory: TheorySchema | None = None

    # --- Backend (at most one) ---
    kwdagger: dict[str, Any] | None = None
    pipeline: dict[str, Any] | None = None

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
