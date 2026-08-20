"""
A dependency-free stand-in for the theory predicates.

Copy this file into your own repository as ``magnet_theory.py`` and annotate
your code with it. Every predicate is a no-op: it returns its target unchanged
and imports nothing beyond the standard library. Your code runs identically
whether or not anyone is auditing it, and you take on no dependency to describe
your own assumptions.

Annotations are collected from your **source**, so none of this has to
execute.

To install a copy::

    python -m magnet.theory.shim --install path/to/your/repo/magnet_theory.py

Usage is the same as the real package::

    from magnet_theory import approximates, assumes, grounds

    # bare -- for a gap that is an *absence* of code
    assumes('Paper.main::hlipschitz', severity='high',
            informal='score smoothness is assumed and never tested')

    # decorator -- on the code that does it
    @approximates('Paper.main::hcover', informal='fixed pool; theory needs density')
    def __init__(self, num_example_runs=64): ...

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
``grounds``      this claim is the empirical shadow of that statement
===============  =========================================================

Reference strings are ``Declaration::binder`` -- the fully-qualified statement
name and the hypothesis binder within it. Binders survive refactors that
invalidate a file and line within weeks.
"""

__all__ = [
    'satisfies',
    'approximates',
    'substitutes',
    'assumes',
    'ignores',
    'violates',
    'grounds',
    'PREDICATES',
]

PREDICATES = (
    'satisfies',
    'approximates',
    'substitutes',
    'assumes',
    'ignores',
    'violates',
    'grounds',
)


class Annotation(object):
    """
    What a predicate returns. Inert.

    Usable bare or as a decorator; decorating returns the object unchanged.
    """

    def __init__(self, predicate, ref, options):
        self.predicate = predicate
        self.ref = ref
        self.options = options

    def __call__(self, obj):
        return obj

    def __repr__(self):
        return '<{} {}>'.format(self.predicate, self.ref)


def _predicate(name):
    def verb(ref, severity=None, **options):
        if severity is not None:
            options['severity'] = severity
        return Annotation(name, ref, options)

    verb.__name__ = name
    return verb


satisfies = _predicate('satisfies')
approximates = _predicate('approximates')
substitutes = _predicate('substitutes')
assumes = _predicate('assumes')
ignores = _predicate('ignores')
violates = _predicate('violates')
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

    # This file is vendored into repositories that do not install MAGNET, so
    # it imports nothing outside the standard library. Hence argparse.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--install', metavar='PATH', help='write a copy here')
    args = parser.parse_args(argv)
    if args.install:
        print('Wrote {}'.format(_install(args.install)))
    else:
        parser.print_help()


if __name__ == '__main__':
    _main()
