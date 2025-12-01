Running HEIM
------------

This document details how to run HEIM benchmarks with the goal of getting CLIP
scores for images generated using prompts on a dataset.

The official docs are here: https://crfm-helm.readthedocs.io/en/latest/heim/


Option 1: Building HEIM Images
==============================

The HEIM dependencies are tricky, so we will use docker. To setup a HEIM aware
image we will first clone this repo and navigate to its root. We will put code
in ``~/code``, but you can do this wherever you would like.

.. code:: bash

   mkdir -p $HOME/code
   cd $HOME/code
   git clone https://github.com/AIQ-Kitware/aiq-magnet.git

   cd $HOME/code/aiq-magnet


Next we will build the docker images.

.. code:: bash

    # Build the UV image
    PUSH_IMAGES=0 \
    PYTHON_VERSION=3.10 \
    UV_VERSION=0.8.8 \
    BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
        ./dockerfiles/uv.build_and_publish.sh

    # Build the magnet image
    PUSH_IMAGES=0 \
    BASE_IMAGE=uv:0.8.8-python3.10-cuda12.4.1-cudnn-devel-ubuntu22.04 \
        ./dockerfiles/magnet.build_and_publish.sh

    # Build the magnet+HEIM image
    PUSH_IMAGES=0 \
    HELM_REMOTE=https://github.com/AIQ-Kitware/helm.git \
    HELM_GIT_REF=explicit_plugins \
    BASE_IMAGE=magnet:latest \
        ./dockerfiles/magnet-heim.build_and_publish.sh

    docker run --rm -it magnet:latest-heim python -c "import helm; print(helm.__file__)" || echo "import helm failed (check install)"


Option 2: Pull Our HEIM Images
==============================

docker pull ghcr.io/aiq-kitware/magnet:latest-heim


Running HEIM
=============

Given that we have a HEIM image, let's ensure we can run it.

We need to specify a data directory on our host system where we can place data
and results. We also want to have a second directory where we can mount custom
configs and plugins, which can be teh same directory, but I'll set it to be
different here.


.. code:: bash

    HOST_DATA_DIRECTORY=/data/reproduce_heim/data
    HOST_CONFIG_DIRECTORY=/data/reproduce_heim/config
    mkdir -p "$HOST_DATA_DIRECTORY"
    mkdir -p "$HOST_CONFIG_DIRECTORY"

    # NOTE: pip install xdev on your local system to run these commands as-is,
    # otherwise just put the text into the correct files with proper
    # indentation.
    uv pip install xdev

    # Write a custom logging config file to our config directory
    xdev codeblock '
    version: 1
    disable_existing_loggers: false

    formatters:
      colored:
        (): colorlog.ColoredFormatter
        #format: "%(bold_black)s%(asctime)s%(reset)s %(log_color)s%(levelname)-8s%(reset)s %(message)s"
        format: "[%(bold_black)s%(asctime)s%(reset)s %(log_color)s%(levelname)-8s%(reset)s %(pathname)s:%(lineno)d %(funcName)s] %(message)s"
        datefmt: "%Y-%m-%dT%H:%M:%S"
        log_colors:
          DEBUG:    cyan
          INFO:     green
          WARNING:  yellow
          ERROR:    red
          CRITICAL: red,bg_white

      standard:
        format: "[%(asctime)s %(levelname)-8s %(pathname)s:%(lineno)d %(funcName)s %(name)s] %(message)s"
        datefmt: "%Y-%m-%dT%H:%M:%S"

    handlers:
      console:
        class: logging.StreamHandler
        level: DEBUG
        formatter: colored
        stream: ext://sys.stdout

      debug_file:
        class: logging.FileHandler
        level: DEBUG
        formatter: standard
        filename: debug.log
        mode: w

      info_file:
        class: logging.FileHandler
        level: INFO
        formatter: standard
        filename: info.log
        mode: w

    loggers:
      helm:
        level: DEBUG
        handlers: [console, debug_file, info_file]
        propagate: false
    ' > $HOST_CONFIG_DIRECTORY/helm_debug_log_config.yaml

    # Write a custom run spec to the config directory
    xdev codeblock '
    from helm.benchmark.run_spec import RunSpec, run_spec_function
    from helm.benchmark.adaptation.adapter_spec import AdapterSpec
    from helm.benchmark.metrics.metric import MetricSpec
    from helm.benchmark.run_specs.classic_run_specs import get_basic_metric_specs
    from helm.benchmark.scenarios.scenario import ScenarioSpec
    from typing import List
    from helm.benchmark.run_specs.heim_run_specs import get_image_generation_adapter_spec


    def get_my_core_heim_metric_specs() -> List[MetricSpec]:
        """Evaluate every image with these set of metrics."""
        return [
            MetricSpec(class_name="helm.benchmark.metrics.image_generation.clip_score_metrics.CLIPScoreMetric", args={}),
        ] + get_basic_metric_specs(names=[])


    @run_spec_function("my_custom_run_spec")
    def get_my_mscoco_spec(
        for_efficiency: bool = False,
        compute_fid: bool = False,
        run_human_eval: bool = False,
        num_human_examples: int = 100,
        use_perturbed: bool = False,
        skip_photorealism: bool = False,
        skip_subject: bool = False,
    ) -> RunSpec:
        scenario_spec = ScenarioSpec(
            class_name="helm.benchmark.scenarios.image_generation.mscoco_scenario.MSCOCOScenario", args={}
        )

        adapter_spec: AdapterSpec
        metric_specs: List[MetricSpec]
        run_spec_name: str

        adapter_spec = get_image_generation_adapter_spec(num_outputs=4)
        metric_specs = get_my_core_heim_metric_specs()
        run_spec_name = "my_custom_run_spec"

        return RunSpec(
            name=run_spec_name,
            scenario_spec=scenario_spec,
            adapter_spec=adapter_spec,
            metric_specs=metric_specs,
            groups=[run_spec_name],
        )
    ' > "$HOST_CONFIG_DIRECTORY"/my_custom_run_spec.py


Now, let's run our custom HEIM spec. You will likely need to have a huggingface
token to access models / data from hugging face. Set this as an environment
variable: ``HF_TOKEN``, something like:


.. code:: bash

    load_secrets
    export HF_TOKEN="$MY_HF_TOKEN"

.. code:: bash

    # Ensure these are the same as above
    HOST_DATA_DIRECTORY=/data/reproduce_heim/data
    HOST_CONFIG_DIRECTORY=/data/reproduce_heim/config

    IMAGE_QUALNAME=magnet:latest-heim
    docker run \
        --rm \
        --gpus=all \
        --workdir /host/data \
        -e HF_TOKEN \
        -v "$HOST_DATA_DIRECTORY":/host/data \
        -v "$HOST_CONFIG_DIRECTORY":/host/config \
        -it "$IMAGE_QUALNAME" \
        helm-run \
            --run-entries my_custom_run_spec:model=huggingface/stable-diffusion-v1-4 \
            --log-config /host/config/helm_debug_log_config.yaml \
            --output-path /host/data/benchmark_output \
            --plugins /host/config/my_custom_run_spec.py \
            --suite my_custom_run_spec \
            --max-eval-instances 20 \
            --num-threads 1

Note several items in the above command:

* We specify our current working directory in the docker command to the data directory so relative path outputs are caught. (Ideally, this is just for safety, but `prod_env` might currently depend onit)

* We specify the output path, so results are written in an area where we can see them on the host system.

* We point at a custom logging config.

* We use our custom --plugins argument to specify a custom run spec file, which is similar to mscoco, but only with CLIP scores. This has not been accepted into standard HELM yet (https://github.com/stanford-crfm/helm/pull/3916).

* We use 20 max-eval-instances and 1 thread to make our benchmark stable and run faster.


When this finishes we can summarize:

.. code:: bash

    # Ensure these are the same as above
    HOST_DATA_DIRECTORY=/data/reproduce_heim/data
    HOST_CONFIG_DIRECTORY=/data/reproduce_heim/config
    IMAGE_QUALNAME=magnet:latest-heim

    # Summarize HELM results
    docker run \
        --rm \
        --gpus=all \
        -e HF_TOKEN \
        --workdir /host/data \
        -v "$HOST_DATA_DIRECTORY":/host/data \
        -v "$HOST_CONFIG_DIRECTORY":/host/config \
        -it "$IMAGE_QUALNAME" \
        helm-summarize \
            --log-config /host/config/helm_debug_log_config.yaml \
            --output-path /host/data/benchmark_output \
            --suite my_custom_run_spec

    # Run the HELM server
    # Note the extra -p argument to publish ports
    docker run \
        --rm \
        --gpus=all \
        -p 8000:8000 \
        -e HF_TOKEN \
        --workdir /host/data \
        -v "$HOST_DATA_DIRECTORY":/host/data \
        -v "$HOST_CONFIG_DIRECTORY":/host/config \
        -it "$IMAGE_QUALNAME" \
        helm-server \
            --output-path /host/data/benchmark_output \
            --suite my_custom_run_spec

Customizing HEIM
================

Optionally we can include our own custom variants of the repos if we need to
develop in the image. This would involve using custom mounts to override the
code repos in the image.

E.g. If you had a custom ``helm`` repo in $HOME/code in your host system, you
could force the image to use it via:

.. code:: bash

    IMAGE_QUALNAME=magnet:latest-heim
    docker run \
        --rm \
        --gpus=all \
        -e HF_TOKEN \
        -v "$HOME/code/helm":/root/code/helm \
        -v "$HOST_DATA_DIRECTORY":/host/data \
        -v "$HOST_CONFIG_DIRECTORY":/host/config \
        -it "$IMAGE_QUALNAME" \
        helm-run --help
