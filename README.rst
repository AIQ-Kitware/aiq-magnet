Developer Quick Start
---------------------

Quick start: install and run tests

.. code:: bash

    uv venv --python 3.11 --seed .venv-311-magnet
    source .venv-311-magnet/bin/activate
    uv pip install .[tests]
    pytest
