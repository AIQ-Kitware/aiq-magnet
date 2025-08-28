"""
Common table helpers (i.e. List[Dict])
"""
import ubelt as ub
import math


class UnhashablePlaceholder(str):
    ...


def _ensure_longform(longform):
    if hasattr(longform, 'to_dicts'):
        # Handle polars
        longform = longform.to_dicts()
    elif hasattr(longform, 'to_dict'):
        # Handle pandas
        longform = longform.to_dict('records')
    return longform


def varied_values(longform, min_variations=0, max_variations=None,
                  default=ub.NoParam, dropna=False, on_error='raise'):
    """
    Given a list of dictionaries, find the values that differ between them.

    Args:
        longform (List[Dict[KT, VT]] | DataFrame):
            This is longform data, as described in [SeabornLongform]_. It is a
            list of dictionaries.

            Each item in the list - or row - is a dictionary and can be thought
            of as an observation. The keys in each dictionary are the columns.
            The values of the dictionary must be hashable. Lists will be
            converted into tuples.

        min_variations (int, default=0):
            "columns" with fewer than ``min_variations`` unique values are
            removed from the result.

        max_variations (int | None):
            If specified only return items with fewer than this number of
            variations.

        default (VT | NoParamType):
            if specified, unspecified columns are given this value.
            Defaults to NoParam.

        on_error (str):
            Error policy when trying to add a non-hashable type.
            Default to "raise". Can be "raise", "ignore", or "placeholder",
            which will impute a hashable error message.

    Returns:
        Dict[KT, List[VT]] :
            a mapping from each "column" to the set of unique values it took
            over each "row". If a column is not specified for each row, it is
            assumed to take a `default` value, if it is specified.

    Raises:
        KeyError: If ``default`` is unspecified and all the rows
            do not contain the same columns.

    References:
        .. [SeabornLongform] https://seaborn.pydata.org/tutorial/data_structure.html#long-form-data

    Example:
        >>> from magnet.utils.util_tables import *  # NOQA
        >>> longform = [
        >>>     {'a': 'on',  'b': 'red'},
        >>>     {'a': 'on',  'b': 'green'},
        >>>     {'a': 'off', 'b': 'blue'},
        >>>     {'a': 'off', 'b': 'black'},
        >>> ]
        >>> varied = varied_values(longform)
        >>> print(f'varied = {ub.urepr(varied, nl=1, sort=1)}')
        varied = {
            'a': {'off', 'on'},
            'b': {'black', 'blue', 'green', 'red'},
        }
    """
    # Enumerate all defined columns
    import numbers

    longform = _ensure_longform(longform)

    columns = set()
    for row in longform:
        if default is ub.NoParam and len(row) != len(columns) and len(columns):
            missing = set(columns).symmetric_difference(set(row))
            raise KeyError((
                'No default specified and not every '
                'row contains columns {}').format(missing))
        columns.update(row.keys())

    cannonical_nan = float('nan')

    # Build up the set of unique values for each column
    varied = ub.ddict(set)
    for row in longform:
        for key in columns:
            value = row.get(key, default)
            if isinstance(value, list):
                value = tuple(value)
            if isinstance(value, numbers.Number) and math.isnan(value):
                if dropna:
                    continue
                else:
                    # Always use a single nan value such that the id check
                    # passes. Otherwise we could end up with a dictionary that
                    # contains multiple nan keys.
                    # References:
                    # .. [SO6441857] https://stackoverflow.com/questions/6441857/nans-as-key-in-dictionaries
                    value = cannonical_nan
            try:
                varied[key].add(value)
            except TypeError as ex:
                if on_error == 'raise':
                    error_note = f'key={key}, {value}={value}'
                    if hasattr(ex, 'add_note'):
                        # Requires python.311 PEP 678
                        ex.add_note(error_note)
                        raise
                    else:
                        raise type(ex)(str(ex) + chr(10) + error_note)
                elif on_error == 'placeholder':
                    varied[key].add(UnhashablePlaceholder(value))
                elif on_error == 'ignore':
                    ...
                else:
                    raise KeyError(on_error)

    # Remove any column that does not have enough variation
    if min_variations > 0:
        for key, values in list(varied.items()):
            if len(values) < min_variations:
                varied.pop(key)

    if max_variations is not None:
        for key, values in list(varied.items()):
            if len(values) > max_variations:
                varied.pop(key)

    return varied


def varied_value_counts(longform, min_variations=0, max_variations=None,
                        default=ub.NoParam, dropna=False, on_error='raise'):
    """
    Given a list of dictionaries, find the values that differ between them.

    Args:
        longform (List[Dict[KT, VT]] | DataFrame):
            This is longform data, as described in [SeabornLongform]_. It is a
            list of dictionaries.

            Each item in the list - or row - is a dictionary and can be thought
            of as an observation. The keys in each dictionary are the columns.
            The values of the dictionary must be hashable. Lists will be
            converted into tuples.

        min_variations (int):
            "columns" with fewer than ``min_variations`` unique values are
            removed from the result. Defaults to 0.

        max_variations (int | None):
            If specified only return items with fewer than this number of
            variations.

        default (VT | NoParamType):
            if specified, unspecified columns are given this value.
            Defaults to NoParam.

        on_error (str):
            Error policy when trying to add a non-hashable type.
            Default to "raise". Can be "raise", "ignore", or "placeholder",
            which will impute a hashable error message.

    Returns:
        Dict[KT, Dict[VT, int]] :
            a mapping from each "column" to the set of unique values it took
            over each "row" and how many times it took that value. If a column
            is not specified for each row, it is assumed to take a `default`
            value, if it is specified.

    Raises:
        KeyError: If ``default`` is unspecified and all the rows
            do not contain the same columns.

    References:
        .. [SeabornLongform] https://seaborn.pydata.org/tutorial/data_structure.html#long-form-data

    Example:
        >>> from magnet.utils.util_tables import *  # NOQA
        >>> longform = [
        >>>     {'a': 'on',  'b': 'red'},
        >>>     {'a': 'on',  'b': 'green'},
        >>>     {'a': 'off', 'b': 'blue'},
        >>>     {'a': 'off', 'b': 'black'},
        >>> ]
        >>> varied_counts = varied_value_counts(longform)
        >>> print(f'varied_counts = {ub.urepr(varied_counts, nl=1, sort=1)}')
        varied_counts = {
            'a': {'off': 2, 'on': 2},
            'b': {'black': 1, 'blue': 1, 'green': 1, 'red': 1},
        }
    """
    # Enumerate all defined columns
    import numbers

    longform = _ensure_longform(longform)

    columns = set()
    for row in longform:
        if default is ub.NoParam and len(row) != len(columns) and len(columns):
            missing = set(columns).symmetric_difference(set(row))
            raise KeyError((
                'No default specified and not every '
                'row contains columns {}').format(missing))
        columns.update(row.keys())

    cannonical_nan = float('nan')

    # Build up the set of unique values for each column
    from collections import Counter
    varied_counts = ub.ddict(Counter)
    for row in longform:
        for key in columns:
            value = row.get(key, default)
            if isinstance(value, list):
                value = tuple(value)

            if isinstance(value, numbers.Number) and math.isnan(value):
                if dropna:
                    continue
                else:
                    # Always use a single nan value such that the id check
                    # passes. Otherwise we could end up with a dictionary that
                    # contains multiple nan keys.
                    # References:
                    # .. [SO6441857] https://stackoverflow.com/questions/6441857/nans-as-key-in-dictionaries
                    value = cannonical_nan
            try:
                varied_counts[key][value] += 1
            except TypeError as ex:
                if on_error == 'raise':
                    error_note = f'key={key}, {value}={value}'
                    if hasattr(ex, 'add_note'):
                        # Requires python.311 PEP 678
                        ex.add_note(error_note)
                        raise
                    else:
                        raise type(ex)(str(ex) + chr(10) + error_note)
                elif on_error == 'placeholder':
                    varied_counts[key][UnhashablePlaceholder(value)] += 1
                elif on_error == 'ignore':
                    ...
                else:
                    raise KeyError(on_error)

    # Remove any column that does not have enough variation
    if min_variations > 0:
        for key, values in list(varied_counts.items()):
            if len(values) < min_variations:
                varied_counts.pop(key)

    if max_variations is not None:
        for key, values in list(varied_counts.items()):
            if len(values) > max_variations:
                varied_counts.pop(key)

    return varied_counts
