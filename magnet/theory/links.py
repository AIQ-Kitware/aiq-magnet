"""Serialized static links between empirical code and theoretical objects."""
from dataclasses import dataclass

PREMISE_SEPARATOR = '::'

__all__ = ['Link', 'PREMISE_SEPARATOR', 'split_ref']


def split_ref(ref: str) -> tuple[str, str | None]:
    """Split ``EntryId`` or ``EntryId::binder`` into its two components."""
    if PREMISE_SEPARATOR not in ref:
        return ref, None
    entry_id, separator, premise_id = ref.partition(PREMISE_SEPARATOR)
    if not separator or not entry_id or not premise_id or PREMISE_SEPARATOR in premise_id:
        raise ValueError(
            f'invalid theory reference {ref!r}; expected EntryId or EntryId::binder'
        )
    return entry_id, premise_id


@dataclass(frozen=True)
class Link:
    """One statically extracted relationship."""

    relation: str
    ref: str
    note: str = ''
    file: str = ''
    line: int | None = None
    qualname: str = ''

    @property
    def target_kind(self) -> str:
        """Whether this link targets a whole statement or one named premise."""
        _, premise_id = split_ref(self.ref)
        return 'premise' if premise_id is not None else 'statement'

    @property
    def entry_id(self) -> str:
        """The parent theory entry ID."""
        return split_ref(self.ref)[0]

    def to_dict(self) -> dict:
        data = {
            'relation': self.relation,
            'ref': self.ref,
        }
        if self.note:
            data['note'] = self.note
        if self.file:
            data['file'] = self.file
        if self.line is not None:
            data['line'] = self.line
        if self.qualname:
            data['qualname'] = self.qualname
        return data
