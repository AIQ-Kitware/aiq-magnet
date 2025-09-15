from helm.common.request import Request
from helm.common.request import RequestResult, GeneratedOutput
from helm.common.authentication import Authentication
from helm.benchmark.executor import ExecutionSpec, Executor, ExecutorError
from helm.benchmark.config_registry import (
    # register_configs_from_directory,
    register_builtin_configs_from_helm_package,
)
from helm.common.hierarchical_logger import hwarn


class HELMInferenceEngine:
    def __init__(self, execution_spec=None):
        # Not sure this is the best place to do this, or if it's
        # idempotent
        register_builtin_configs_from_helm_package()

        if execution_spec is None:
            auth = Authentication("")
            url = None
            local_path = "prod_env"
            num_threads = 1
            dry_run = False
            sqlite_cache_backend_config = None
            mongo_cache_backend_config = None

            execution_spec = ExecutionSpec(
                auth=auth,
                url=url,
                local_path=local_path,
                parallelism=num_threads,
                dry_run=dry_run,
                sqlite_cache_backend_config=sqlite_cache_backend_config,
                mongo_cache_backend_config=mongo_cache_backend_config)

        self.executor = Executor(execution_spec)

    def inference_request(self, request: Request) -> RequestResult:
        # Largely copied (with a few tweaks to remove the RequestState
        # container) from the Executor.process method in HELM (
        # https://github.com/stanford-crfm/helm/blob/v0.5.8/src/helm/benchmark/executor.py#L111)
        try:
            result: RequestResult = self.executor.context.make_request(request)
        except Exception as e:
            raise ExecutorError(f"{str(e)} Request: {request}") from e
        if not result.success:
            if result.error_flags and not result.error_flags.is_fatal:
                hwarn(f"Non-fatal error treated as empty completion: {result.error}")
                result.completions = [GeneratedOutput(text="", logprob=0, tokens=[])]
            else:
                raise ExecutorError(f"{str(result.error)} Request: {request}")
        return result
