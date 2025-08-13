#!/bin/bash
__doc__="
Generate xcookie-style CI scripts
"
cd "$HOME"/code/magnet-sys-exploratory
xcookie --only_gen ".github*tests.yml" --enable_gpg=False --ci_pypy_versions="" --max_python="3.13" \
    --use_pyproject_requirements=True \
    --os=linux \
    --ci_versions_minimal_strict=None \
    --ci_versions_minimal_loose=None \
    --linter=False \
    --deploy_pypi=False

