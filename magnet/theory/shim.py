"""
A dependency-free stand-in for the theory predicates.

Copy this file into your own repository as ``magnet_theory.py`` and annotate
your code with it. Every predicate here is a no-op: it returns its target
unchanged, works as a decorator, a context manager, or a bare call, and imports
nothing beyond the standard library. Your code runs identically whether or not
anyone is auditing it, and you take on no dependency to describe your own
assumptions.

The annotations are read from your **source**, not from your imports, so none
of this has to execute for the assumptions to be collected.

To install a copy::

    python -m magnet.theory.shim --install path/to/your/repo/magnet_theory.py

Usage is the same as the real package::

    from magnet_theory import approximates, assumes, grounds, substitutes

    # (a) bare -- for a gap that is an *absence* of code
    assumes('Paper.main::hlipschitz', severity='high',
            informal='score smoothness is assumed and never tested')

    # (b) decorator -- on the code that does it
    @approximates('Paper.main::hcover', informal='fixed pool; theory needs density')
    def __init__(self, num_example_runs=64): ...

    # (c) context manager -- around a region, with values recorded
    with substitutes('Paper.main::hpsi', kind='different-object') as edge:
        coords = embed(texts)
        edge.witness(embedder='some-embedding-model')

    # on the claim itself
    @grounds('Paper.TheoryPractice.EmpiricalCrossBudgetMAEClaim')
    def claim(results): ...

The predicates, and what each asserts about the code it annotates:

===============  =========================================================
``satisfies``    the experiment establishes the hypothesis
``approximates`` the same object, in a finite or numerical version
``substitutes``  a *different* object stands in for the theorem's
``assumes``      relied on; nothing establishes or checks it
``ignores``      a side condition delimiting the regime, dropped
``violates``     known to fail; pass ``evidence=`` if there is a proof
``checks``       tested at run time; the decorated function returns the result
``grounds``      this claim is the empirical shadow of that statement
===============  =========================================================

Reference strings are ``Declaration::binder`` -- the fully-qualified statement
name and the hypothesis binder within it. Use binder names, not file and line:
line numbers go stale within weeks, binders survive.
"""
import functools
import inspect

__all__ = [
    'satisfies',
    'approximates',
    'substitutes',
    'assumes',
    'ignores',
    'violates',
    'checks',
    'grounds',
    'Check',
    'Annotation',
]

PREDICATES = (
    'satisfies',
    'approximates',
    'substitutes',
    'assumes',
    'ignores',
    'violates',
    'checks',
    'grounds',
)


class Check(object):
    """Result of a runtime hypothesis check. Inert here; recorded by MAGNET."""

    __slots__ = ('ok', 'value', 'detail')

    def __init__(self, ok, value=None, detail=''):
        self.ok = bool(ok)
        self.value = value
        self.detail = detail

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return 'Check(ok=%r, value=%r)' % (self.ok, self.value)


class Annotation(object):
    """
    An inert annotation that activates three ways.

    Deliberately does nothing. The real implementation records observations and
    witnessed values; this one keeps the same shape so annotated code behaves
    identically with either installed.
    """

    __slots__ = ('predicate', 'ref', 'options')

    def __init__(self, predicate, ref, options):
        self.predicate = predicate
        self.ref = ref
        self.options = options

    # decorator
    def __call__(self, obj):
        if not (inspect.isfunction(obj) or inspect.ismethod(obj)):
            return obj

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            return obj(*args, **kwargs)

        return wrapper

    # context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def witness(self, **values):
        """Record observed values. Inert here."""
        return self

    def record_check(self, result):
        """Record a check result. Inert here."""
        return result

    def __repr__(self):
        return '<%s %s>' % (self.predicate, self.ref)


def _predicate(name):
    def verb(ref, severity=None, **options):
        if severity is not None:
            options['severity'] = severity
        return Annotation(name, ref, options)

    verb.__name__ = name
    verb.__qualname__ = name
    verb.__doc__ = 'Inert %r annotation; see the module docstring.' % (name,)
    return verb


satisfies = _predicate('satisfies')
approximates = _predicate('approximates')
substitutes = _predicate('substitutes')
assumes = _predicate('assumes')
ignores = _predicate('ignores')
violates = _predicate('violates')
checks = _predicate('checks')
grounds = _predicate('grounds')


def _install(destination):
    """Copy this file to ``destination``."""
    import os
    import shutil

    source = os.path.abspath(__file__)
    destination = os.path.abspath(destination)
    if os.path.isdir(destination):
        destination = os.path.join(destination, 'magnet_theory.py')
    if source == destination:
        raise ValueError('refusing to copy the shim over itself')
    shutil.copyfile(source, destination)
    return destination


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog='python -m magnet.theory.shim',
        description='Install the dependency-free theory-annotation shim.',
    )
    parser.add_argument(
        '--install',
        metavar='PATH',
        help='destination file or directory; a directory gets magnet_theory.py',
    )
    args = parser.parse_args(argv)
    if args.install:
        print('Wrote %s' % (_install(args.install),))
    else:
        parser.print_help()


if __name__ == '__main__':
    _main()
