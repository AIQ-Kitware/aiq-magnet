#!/bin/bash
__doc__="
Generate xcookie-style CI scripts
"
cd "$HOME"/code/aiq-magnet
xcookie --only_gen ".github*.yml" --enable_gpg=False --ci_pypy_versions="" --max_python="3.13" \
    --use_pyproject_requirements=True \
    --os=linux \
    --mod_name=magnet \
    --pkg_name=aiq-magnet \
    --ci_versions_minimal_strict=None \
    --ci_versions_minimal_loose=None \
    --linter=False \
    --ci_pypi_trusted_publishing=True \
    --deploy_pypi=True


xcookie --enable_gpg=False --ci_pypy_versions="" --max_python="3.13" \
    --use_pyproject_requirements=True \
    --os=linux \
    --mod_name=magnet \
    --pkg_name=aiq-magnet \
    --ci_versions_minimal_strict=None \
    --ci_versions_minimal_loose=None \
    --linter=False \
    --ci_pypi_trusted_publishing=True \
    --deploy_pypi=True


xcookie --only_gen "pyproject*" --enable_gpg=False --ci_pypy_versions="" --max_python="3.13" \
    --use_pyproject_requirements=True \
    --os=linux \
    --mod_name=magnet \
    --pkg_name=aiq-magnet \
    --ci_versions_minimal_strict=None \
    --ci_versions_minimal_loose=None \
    --linter=False \
    --deploy_pypi=False --regen pyproj

