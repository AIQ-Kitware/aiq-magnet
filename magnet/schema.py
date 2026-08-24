from enum import StrEnum
from typing import Any, Literal, Optional
from pydantic import AliasChoices, BaseModel, Field, model_validator
from pydantic import ConfigDict

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


class EvidenceSchema(BaseModel):
    """Selection policy for KWDagger aggregate rows used as evidence."""

    model_config = ConfigDict(extra='forbid')

    scope: Literal['all', 'requested'] = 'all'


class KWDaggerSchema(BaseModel):
    """
    A card's kwdagger backend.

    Everything but ``result_node`` is passed as the ``params`` payload to
    ``kwdagger schedule``. Unknown keys are allowed so KWDagger can own its
    matrix/configuration language while MAGNET validates only what it reads.
    """
    model_config = ConfigDict(extra='allow')

    #: The node whose accumulated aggregate rows provide claim evidence.
    result_node: str | None = None

    #: A Pipeline callable, a path to a declarative pipeline, or the pipeline
    #: inline as a mapping.
    pipeline: str | dict[str, Any]

    matrix: dict[str, Any] | None = None

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
    # `depends` is an alias for `depends_on`.
    # TODO: modify "depends_on" to reference an actual symbol
    depends_on: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices('depends_on', 'depends'),
    )
    python: str | None = None # TODO: this can be validated with a syntax check
    metadata: SymbolMetadataSchema | None = None

    @model_validator(mode='before')
    @classmethod
    def dependency_aliases_agree(cls, data: Any) -> Any:
        """
        Error both depends_on and depends are given and they disagree.
        """
        if isinstance(data, dict):
            depends_on = data.get('depends_on')
            depends = data.get('depends')
            if (
                depends_on is not None
                and depends is not None
                and depends_on != depends
            ):
                raise ValueError('`depends_on` and `depends` must agree')
        return data

    @model_validator(mode='after')
    def has_resolution(self) -> 'SymbolSchema':
        if (
            self.type is None
            and self.value is None
            and self.sweep is None
            and self.python is None
        ):
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

    # --- Backend (at most one) ---
    kwdagger: KWDaggerSchema | None = None
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


class NewEvaluationKWDaggerSchema(KWDaggerSchema):
    """KWDagger block required by the replacement evaluation API."""

    result_node: str


class NewEvaluationRecipeSchema(EvaluationCardSchema):
    """Schema for a recipe consumed by ``magnet evaluate_new``."""

    kwdagger: NewEvaluationKWDaggerSchema
    pipeline: None = None
    evidence: EvidenceSchema = Field(default_factory=EvidenceSchema)

    @model_validator(mode='after')
    def no_legacy_symbol_sweeps(self) -> 'NewEvaluationRecipeSchema':
        sweep_symbols = sorted(
            name
            for name, symbol in (self.symbols or {}).items()
            if symbol.sweep is not None
        )
        if sweep_symbols:
            raise ValueError(
                'evaluate_new does not execute legacy symbol sweeps; move '
                'experimental variation into `kwdagger.matrix`. Sweep '
                f'symbols: {sweep_symbols}'
            )
        return self
