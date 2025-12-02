#!/usr/bin/env python
if __name__ == '__main__':
    import pytest
    import sys
    mod_name = 'magnet'
    mod_dpath = 'magnet'
    test_dpath = 'tests'
    pytest_args = [
        '--cov-config', 'pyproject.toml',
        '--cov-report', 'html',
        '--cov-report', 'term',
        '--xdoctest',
        '--cov=' + mod_name,
        mod_dpath, test_dpath
    ]
    pytest_args = pytest_args + sys.argv[1:]
    sys.exit(pytest.main(pytest_args))
