# magnet-theory

`magnet-theory` is the dependency-free annotation vocabulary used by MAGNET to
connect empirical Python code to theoretical statements and named premises.
The decorators and context managers are runtime no-ops, so application code can
annotate itself without depending on the full `aiq-magnet` stack.

```python
import magnet_theory as theory

@theory.tests('Examples.Stability.Theorem')
@theory.assumes('Examples.Stability.Theorem::hiid')
def experiment():
    ...
```

Install only the annotation package with:

```bash
pip install magnet-theory
```

The full `aiq-magnet` distribution contains the static parser, theory indexes,
validation, reporting, and evaluation integration that interpret these
annotations.
