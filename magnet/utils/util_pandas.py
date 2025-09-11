"""
Heavilly modified / simplified subset of data frame extensions ported from geowatch
"""
import ubelt as ub
import math
import pandas as pd


class DataFrame(pd.DataFrame):
    """
    Extension of pandas dataframes with quality-of-life improvements.

    Refernces:
        .. [SO22155951] https://stackoverflow.com/questions/22155951/how-can-i-subclass-a-pandas-dataframe

    Example:
        >>> from magnet.utils.util_pandas import *  # NOQA
        >>> from magnet.utils import util_pandas
        >>> df = util_pandas.DataFrame.random(rng=0, rows=2, columns='ab')
        >>> print(df)
                  a         b
        0  0.548814  0.715189
        1  0.602763  0.544883
    """
    @property
    def _constructor(self):
        return DataFrame

    @classmethod
    def random(cls, rows=10, columns='abcde', rng=None):
        """
        Create a random data frame for testing.
        """
        import kwarray
        rng = kwarray.ensure_rng(rng)

        def coerce_index(data):
            if isinstance(data, int):
                return list(range(data))
            else:
                return list(data)
        columns = coerce_index(columns)
        index = coerce_index(rows)
        random_data = [{c: rng.rand() for c in columns} for r in index]
        self = cls(random_data, index=index, columns=columns)
        return self

    @classmethod
    def coerce(cls, data):
        """
        Ensures that the input is an instance of our extended DataFrame.

        Pandas is generally good about input coercion via its normal
        constructors, the purpose of this classmethod is to quickly ensure that
        a DataFrame has all of the extended methods defined by this class
        without incurring a copy. In this sense it is more similar to
        :func:numpy.asarray`.

        Args:
            data (DataFrame | ndarray | Iterable | dict):
                generally another dataframe, otherwise normal inputs that would
                be given to the regular pandas dataframe constructor

        Returns:
            DataFrame:

        Example:
            >>> # xdoctest: +REQUIRES(--benchmark)
            >>> # This example demonstrates the speed difference between
            >>> # recasting as a DataFrame versus using coerce
            >>> from magnet.utils.util_pandas import DataFrame
            >>> data = DataFrame.random(rows=10_000)
            >>> import timerit
            >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
            >>> for timer in ti.reset('constructor'):
            >>>     with timer:
            >>>         DataFrame(data)
            >>> for timer in ti.reset('coerce'):
            >>>     with timer:
            >>>         DataFrame.coerce(data)
            >>> # xdoctest: +IGNORE_WANT
            Timed constructor for: 100 loops, best of 10
                time per loop: best=2.594 µs, mean=2.783 ± 0.1 µs
            Timed coerce for: 100 loops, best of 10
                time per loop: best=246.000 ns, mean=283.000 ± 32.4 ns
        """
        if isinstance(data, cls):
            return data
        else:
            return cls(data)

    def safe_drop(self, labels, axis=0):
        """
        Like :func:`self.drop`, but does not error if the specified labels do
        not exist.

        Args:
            df (pd.DataFrame): df
            labels (List): ...
            axis (int): todo

        Example:
            >>> from magnet.utils.util_pandas import *  # NOQA
            >>> import numpy as np
            >>> self = DataFrame({k: np.random.rand(10) for k in 'abcde'})
            >>> self.safe_drop(list('bdf'), axis=1)
        """
        existing = self.axes[axis]
        labels = existing.intersection(labels)
        return self.drop(labels, axis=axis)

    def reorder(self, head=None, tail=None, axis=0, missing='error',
                fill_value=float('nan'), **kwargs):
        """
        Change the order of the row or column index. Unspecified labels will
        keep their existing order after the specified labels.

        Args:
            head (List | None):
                The order of the labels to put at the start of the re-indexed
                data frame. Unspecified labels keep their relative order and
                are placed after specified these "head" labels.

            tail (List | None):
                The order of the labels to put at the end of the re-indexed
                data frame. Unspecified labels keep their relative order and
                are placed after before these "tail" labels.

            axis (int):
                The axis 0 for rows, 1 for columns to reorder.

            missing (str):
                Policy to handle specified labels that do not exist in the
                specified axies. Can be either "error", "drop", or "fill".
                If "drop", then drop any specified labels that do not exist.
                If "error", then raise an error non-existing labels are given.
                If "fill", then fill in values for labels that do not exist.

            fill_value (Any):
                fill value to use when missing is "fill".

        Returns:
            Self - DataFrame with modified indexes

        Example:
            >>> from magnet.utils import util_pandas
            >>> self = util_pandas.DataFrame.random(rows=5, columns=['a', 'b', 'c', 'd', 'e', 'f'])
            >>> new = self.reorder(['b', 'c'], axis=1)
            >>> assert list(new.columns) == ['b', 'c', 'a', 'd', 'e', 'f']
            >>> # Set the order of the first and last of the columns
            >>> new = self.reorder(head=['b', 'c'], tail=['e', 'd'], axis=1)
            >>> assert list(new.columns) == ['b', 'c', 'a', 'f', 'e', 'd']
            >>> # Test reordering the rows
            >>> new = self.reorder([1, 0], axis=0)
            >>> assert list(new.index) == [1, 0, 2, 3, 4]
            >>> # Test reordering with a non-existent column
            >>> new = self.reorder(['q'], axis=1, missing='drop')
            >>> assert list(new.columns) == ['a', 'b', 'c', 'd', 'e', 'f']
            >>> new = self.reorder(['q'], axis=1, missing='fill')
            >>> assert list(new.columns) == ['q', 'a', 'b', 'c', 'd', 'e', 'f']
            >>> import pytest
            >>> with pytest.raises(ValueError):
            >>>     self.reorder(['q'], axis=1, missing='error')
            >>> # Should error if column is given in both head and tail
            >>> with pytest.raises(ValueError):
            >>>     self.reorder(['c'], ['c'], axis=1, missing='error')
        """
        if 'intersect' in kwargs:
            raise Exception('The intersect argument was removed. Set missing=drop')
        if kwargs:
            raise ValueError(f'got unknown kwargs: {list(kwargs.keys())}')

        existing = self.axes[axis]
        if head is None:
            head = []
        if tail is None:
            tail = []
        head_set = set(head)
        tail_set = set(tail)
        duplicate_labels = head_set & tail_set
        if duplicate_labels:
            raise ValueError(
                'Cannot specify the same label in both the head and tail.'
                f'Duplicate labels: {duplicate_labels}')
        if missing == 'drop':
            orig_order = ub.oset(list(existing))
            resolved_head = ub.oset(head) & orig_order
            resolved_tail = ub.oset(tail) & orig_order
        elif missing == 'error':
            requested = (head_set | tail_set)
            unknown = requested - set(existing)
            if unknown:
                raise ValueError(
                    f"Requested labels that don't exist unknown={unknown}. "
                    "Specify intersect=True to ignore them.")
            resolved_head = head
            resolved_tail = tail
        elif missing == 'fill':
            resolved_head = head
            resolved_tail = tail
        else:
            raise KeyError(missing)
        remain = existing.difference(resolved_head).difference(resolved_tail)
        new_labels = list(resolved_head) + list(remain) + list(resolved_tail)
        return self.reindex(labels=new_labels, axis=axis,
                            fill_value=fill_value)

    def match_columns(self, pat, hint='glob'):
        """
        Find matching columns in O(N)
        """
        from kwutil import util_pattern
        pat = util_pattern.Pattern.coerce(pat, hint=hint)
        found = [c for c in self.columns if pat.match(c)]
        return found

    def search_columns(self, pat, hint='glob'):
        """
        Find matching columns in O(N)
        """
        from kwutil import util_pattern
        pat = util_pattern.Pattern.coerce(pat, hint=hint)
        found = [c for c in self.columns if pat.search(c)]
        return found

    def varied_values(self, min_variations=0, max_variations=None,
                      default=ub.NoParam, dropna=False, on_error='raise'):
        """
        Summarize how which values are varied within each column

        Kwargs:
            min_variations=0, max_variations=None, default=ub.NoParam,
            dropna=False, on_error='raise'

        Example:
            >>> from magnet.utils.util_pandas import DataFrame
            >>> self = (DataFrame.random(rows=5, rng=0) * 5).round().astype(int)
            >>> print(self)
               a  b  c  d  e
            0  3  4  3  3  2
            1  3  2  4  5  2
            2  4  3  3  5  0
            3  0  0  4  4  4
            4  5  4  2  4  1
            >>> varied = self.varied_values()
            >>> print(f'varied = {ub.urepr(varied, nl=1, sort=1)}')
            varied = {
                'a': {0, 3, 4, 5},
                'b': {0, 2, 3, 4},
                'c': {2, 3, 4},
                'd': {3, 4, 5},
                'e': {0, 1, 2, 4},
            }

        """
        from magnet.utils.util_tables import varied_values
        varied = varied_values(
            self, min_variations=min_variations,
            max_variations=max_variations,
            default=default, dropna=dropna, on_error=on_error,
        )
        return varied

    def varied_value_counts(self, **kwargs):
        """
        Summarize how many times values are varied within each column

        Kwargs:
            min_variations=0, max_variations=None, default=ub.NoParam,
            dropna=False, on_error='raise'

        Example:
            >>> from magnet.utils.util_pandas import DataFrame
            >>> self = (DataFrame.random(rows=5, rng=0) * 5).round().astype(int)
            >>> varied_counts = self.varied_value_counts()
            >>> print(f'varied_counts = {ub.urepr(varied_counts, nl=1, sort=1)}')
            varied_counts = {
                'a': {0: 1, 3: 2, 4: 1, 5: 1},
                'b': {0: 1, 2: 1, 3: 1, 4: 2},
                'c': {2: 1, 3: 2, 4: 2},
                'd': {3: 1, 4: 2, 5: 2},
                'e': {0: 1, 1: 1, 2: 2, 4: 1},
            }
        """
        from magnet.utils.util_tables import varied_value_counts
        varied_counts = varied_value_counts(self, **kwargs)
        return varied_counts

    def shorten_columns(self, return_mapping=False, min_length=0):
        """
        Shorten column names by separating unique suffixes based on the "."
        separator.

        Args:
            return_mapping (bool):
                if True, returns the

            min_length (int):
                minimum size of the new column names in terms of parts.

        Returns:
            DataFrame | Tuple[DataFrame, Dict[str, str]]:
                Either the new data frame with shortened column names or that
                data frame and the mapping from old column names to new column
                names.

        Example:
            >>> from magnet.utils.util_pandas import DataFrame
            >>> # If all suffixes are unique, then they are used.
            >>> self = DataFrame.random(2, columns=['id', 'params.metrics.f1',
            >>>                                     'params.metrics.acc',
            >>>                                     'params.fit.model.lr',
            >>>                                     'params.fit.data.seed'], rng=0)
            >>> print(self)  # xdoctest: +IGNORE_WANT
                     id  params.metrics.f1  params.metrics.acc  params.fit.model.lr  params.fit.data.seed
            0  0.548814           0.715189            0.602763             0.544883              0.423655
            1  0.645894           0.437587            0.891773             0.963663              0.383442
            >>> new = self.shorten_columns()
            >>> print(new)  # xdoctest: +IGNORE_WANT
                     id        f1       acc        lr      seed
            0  0.026911  0.284601  0.822176  0.194262  0.934294
            1  0.739338  0.286664  0.005998  0.802884  0.557565
            >>> assert list(new.columns) == ['id', 'f1', 'acc', 'lr', 'seed']

        Example:
            >>> from magnet.utils.util_pandas import DataFrame
            >>> # Conflicting suffixes impose limitations on what can be shortened
            >>> self = DataFrame.random(2, columns=['id', 'params.metrics.magic',
            >>>                                     'params.metrics.acc',
            >>>                                     'params.fit.model.lr',
            >>>                                     'params.fit.data.magic'],
            >>>                         rng=0)
            >>> new = self.shorten_columns()
            >>> print(self)  # xdoctest: +IGNORE_WANT
                     id  params.metrics.magic  params.metrics.acc  params.fit.model.lr  params.fit.data.magic
            0  0.548814              0.715189            0.602763             0.544883               0.423655
            1  0.645894              0.437587            0.891773             0.963663               0.383442
            >>> print(new)  # xdoctest: +IGNORE_WANT
                     id  metrics.magic  metrics.acc  model.lr  data.magic
            0  0.548814       0.715189     0.602763  0.544883    0.423655
            1  0.645894       0.437587     0.891773  0.963663    0.383442
            >>> assert list(new.columns) == ['id', 'metrics.magic', 'metrics.acc', 'model.lr', 'data.magic']
        """
        import ubelt as ub
        from magnet.utils.util_stringalgo import shortest_unique_suffixes
        old_cols = self.columns
        new_cols = shortest_unique_suffixes(old_cols, sep='.', min_length=min_length)
        mapping = ub.dzip(old_cols, new_cols)
        new = self.rename(columns=mapping)
        if return_mapping:
            return new, mapping
        else:
            return new

    def argextrema(self, columns, objective='maximize', k=1):
        """
        Finds the top K indexes (locs) for given columns.

        Args:
            columns (str | List[str]) : columns to find extrema of.
                If multiple are given, then secondary columns are used as
                tiebreakers.

            objective (str | List[str]) :
                Either maximize or minimize (max and min are also accepted).
                If given as a list, it specifies the criteria for each column,
                which allows for a mix of maximization and minimization.

            k : number of top entries

        Returns:
            List: indexes into subset of data that are in the top k for any of the
                requested columns.

        Example:
            >>> from magnet.utils.util_pandas import DataFrame
            >>> # If all suffixes are unique, then they are used.
            >>> self = DataFrame.random(columns=['id', 'f1', 'loss'], rows=10)
            >>> self.loc[3, 'f1'] = 1.0
            >>> self.loc[4, 'f1'] = 1.0
            >>> self.loc[5, 'f1'] = 1.0
            >>> self.loc[3, 'loss'] = 0.2
            >>> self.loc[4, 'loss'] = 0.3
            >>> self.loc[5, 'loss'] = 0.1
            >>> columns = ['f1', 'loss']
            >>> k = 4
            >>> top_indexes = self.argextrema(columns=columns, k=k, objective=['max', 'min'])
            >>> assert len(top_indexes) == k
            >>> print(self.loc[top_indexes])
        """
        ascending = None
        def rectify_ascending(objective_str):
            if objective_str in {'max', 'maximize'}:
                ascending = False
            elif objective_str in {'min', 'minimize'}:
                ascending = True
            else:
                raise KeyError(objective)
            return ascending

        if isinstance(objective, str):
            ascending = rectify_ascending(objective)
        else:
            ascending = [rectify_ascending(o) for o in objective]

        ranked_data = self.sort_values(columns, ascending=ascending)
        if isinstance(k, float) and math.isinf(k):
            k = None
        top_locs = ranked_data.index[0:k]
        return top_locs


class DotDictDataFrame(DataFrame):
    """
    A proof-of-concept wrapper around pandas that lets us walk down the nested
    structure a little easier.

    SeeAlso:
        kwutil.DotDict

    Example:
        >>> # documentation version
        >>> from magnet.utils.util_pandas import *  # NOQA
        >>> rows = [
        >>>     {'node1.id': 1, 'node2.id': 2, 'node1.metrics.ap': 0.5, 'node2.metrics.ap': 0.8},
        >>>     {'node1.id': 1, 'node2.id': 2, 'node1.metrics.ap': 0.5, 'node2.metrics.ap': 0.8},
        >>> ]
        >>> self = DotDictDataFrame(rows)
        >>> print(self)
           node1.id  node2.id  node1.metrics.ap  node2.metrics.ap
        0         1         2               0.5               0.8
        1         1         2               0.5               0.8
        >>> # Lookup by prefix
        >>> print(self.prefix['node1'])
           node1.id  node1.metrics.ap
        0         1               0.5
        1         1               0.5
        >>> # Lookup by suffix
        >>> print(self.suffix['id'])
           node1.id  node2.id
        0         1         2
        1         1         2
        >>> # Lookup by prefix (dropping the prefix)
        >>> print(self.prefix_subframe('node1', drop_prefix=True))
           id  metrics.ap
        0   1         0.5
        1   1         0.5
        >>> # alternative way to get a very concise unambiguous dataframe
        >>> print(self.prefix['node1'].shorten_columns())
           id   ap
        0   1  0.5
        1   1  0.5
    """
    @property
    def _constructor(self):
        return DotDictDataFrame

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def _prefix_columns(self, prefix, with_mapping=False):
        if isinstance(prefix, str):
            prefix_set = {prefix}
            prefixes = (prefix + '.',)
        else:
            prefix_set = set(prefix)
            prefixes = tuple(p + '.' for p in prefix)
        cols = [c for c in self.columns if c.startswith(prefixes) or c in prefix_set]
        mapping = None
        if with_mapping:
            mapping = {}
            for c in cols:
                for p in prefix_set:
                    if c == p or c.startswith(p + '.'):
                        mapping[c] = c[len(p) + 1:]
        return cols, mapping

    def _suffix_columns(self, suffix):
        if isinstance(suffix, str):
            suffix_set = {suffix}
            suffixes = ('.' + suffix,)
        else:
            suffix_set = set(suffix)
            suffixes = tuple('.' + s for s in suffix)
        cols = [c for c in self.columns if c.endswith(suffixes) or c in suffix_set]
        return cols

    def prefix_subframe(self, prefix, drop_prefix=False):
        """
        Get a subset of columns by prefix

        Args:
            prefix (str | List[str]):
                one or more prefixes to lookup

            drop_prefix (bool):
                if True, drop prefixes. Note: this can be ambiguous if mulitple
                prefixes are given.

        Returns:
            DotDictDataFrame

        Example:
            >>> from magnet.utils.util_pandas import *  # NOQA
            >>> part1 = (DotDictDataFrame.random(rows=2, columns='abc', rng=0) * 10).astype(int)
            >>> part2 = (DotDictDataFrame.random(rows=2, columns='abc', rng=0) * 10).astype(int)
            >>> self = pd.concat([part1.insert_prefix('p1'), part2.insert_prefix('p2')], axis=1)
            >>> print(self)
               p1.a  p1.b  p1.c  p2.a  p2.b  p2.c
            0     5     7     6     5     7     6
            1     5     4     6     5     4     6
            >>> new = self.prefix_subframe('p2', drop_prefix=True)
            >>> print(new)
               a  b  c
            0  5  7  6
            1  5  4  6
        """
        if isinstance(prefix, str):
            prefix = [prefix]
        cols, mapping = self._prefix_columns(prefix, with_mapping=drop_prefix)
        new = self.loc[:, cols]
        if drop_prefix:
            new.rename(mapping, inplace=True, axis=1)
        return new

    def suffix_subframe(self, suffix):
        """
        Get a subset of columns by suffix

        Args:
            suffix (str | List[str]):
                one or more prefixes to lookup

        Returns:
            DotDictDataFrame
        """
        cols = self._suffix_columns(suffix)
        new = self.loc[:, cols]
        return new

    @property
    def prefix(self):
        """
        Allows for self.prefix[text] syntax

        SeeAlso:
            DotDictDataFrame.prefix_subframe

        Example:
            >>> from magnet.utils.util_pandas import *  # NOQA
            >>> part1 = (DotDictDataFrame.random(rows=2, columns='abc', rng=0) * 10).astype(int)
            >>> part2 = (DotDictDataFrame.random(rows=2, columns='abc', rng=0) * 10).astype(int)
            >>> self = pd.concat([part1.insert_prefix('p1'), part2.insert_prefix('p2')], axis=1)
            >>> print(self)
               p1.a  p1.b  p1.c  p2.a  p2.b  p2.c
            0     5     7     6     5     7     6
            1     5     4     6     5     4     6
            >>> new = self.prefix['p2']
            >>> print(new)
               p2.a  p2.b  p2.c
            0     5     7     6
            1     5     4     6
        """
        return _PrefixLocIndexer(self)

    @property
    def suffix(self):
        """
        Allows for self.suffix[text] syntax

        SeeAlso:
            DotDictDataFrame.suffix_subframe

        Example:
            >>> from magnet.utils.util_pandas import *  # NOQA
            >>> part1 = (DotDictDataFrame.random(rows=2, columns='abc', rng=0) * 10).astype(int)
            >>> part2 = (DotDictDataFrame.random(rows=2, columns='abc', rng=0) * 10).astype(int)
            >>> self = pd.concat([part1.insert_prefix('p1'), part2.insert_prefix('p2')], axis=1)
            >>> print(self)
               p1.a  p1.b  p1.c  p2.a  p2.b  p2.c
            0     5     7     6     5     7     6
            1     5     4     6     5     4     6
            >>> new = self.suffix['a']
            >>> print(new)
               p1.a  p2.a
            0     5     5
            1     5     5
        """
        return _SuffixLocIndexer(self)

    def insert_prefix(self, prefix):
        """
        Args:
            prefix (str): prefix to insert in all columns with a dot separator

        Returns:
            DotDictDataFrame

        Example:
            >>> from magnet.utils.util_pandas import *  # NOQA
            >>> self = (DotDictDataFrame.random(rows=2, columns='abc', rng=0) * 10).astype(int)
            >>> self = self.insert_prefix('custom')
            >>> print(self)
               custom.a  custom.b  custom.c
            0         5         7         6
            1         5         4         6
        """
        assert not prefix.endswith('.'), 'dont include the dot'
        mapper = {c: prefix + '.' + c for c in self.columns}
        new = self.rename(mapper, axis=1)
        return new


class _PrefixLocIndexer:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, index):
        return self.parent.prefix_subframe(index)


class _SuffixLocIndexer:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, index):
        return self.parent.suffix_subframe(index)
