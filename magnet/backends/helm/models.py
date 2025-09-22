from helm.benchmark.model_deployment_registry import ALL_MODEL_DEPLOYMENTS
from helm.benchmark.config_registry import register_builtin_configs_from_helm_package
from helm.benchmark.model_metadata_registry import ModelMetadata, MODEL_NAME_TO_MODEL_METADATA, UNSUPPORTED_MODEL_TAG, DEPRECATED_MODEL_TAG

class HELMModels:
    """
    Simple collection of ModelDeployments in HELM with corresponding ModelMetadata.


    Collects all available models, model metadata, and tokenizers from HELM package (helm/config/*.yaml).

    Example:
        >>> from magnet.backends.helm.models import HELMModels
        >>> models = HELMModels()
        >>> len(models) # subject to change
        482
    """
    def __init__(self):
        if not ALL_MODEL_DEPLOYMENTS:
            # only run once to avoid duplicates
            register_builtin_configs_from_helm_package()

        self.models = ALL_MODEL_DEPLOYMENTS
        
        self.model_metadata = {
            model: metadata
            for model, metadata in MODEL_NAME_TO_MODEL_METADATA.items()
            if not (UNSUPPORTED_MODEL_TAG in metadata.tags or DEPRECATED_MODEL_TAG in metadata.tags)
        }

    def get_metadata_from_model_name(self, model_name: str) -> ModelMetadata:
        return self.model_metadata[model_name]

    def __len__(self):
        """
        Retrieve number of unique supported models that have metadata
        """
        # models.model_name includes 2 deprecated models
        return len(self.model_metadata)