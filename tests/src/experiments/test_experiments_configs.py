from dataclasses import asdict

from src.analysis.prompt_filterations import AllPromptFilteration
from src.core.names import DatasetName
from src.core.types import MODEL_ARCH, FeatureCategory, TCodeVersionName, TModelSize, TokenType, TWindowSize
from src.experiments.infrastructure.base_runner import (
    BaseVariantParams,
    InputParams,
    MetadataParams,
)
from src.experiments.runners.evaluate_model import EvaluateModelParams, EvaluateModelRunner
from src.experiments.runners.heatmap import HeatmapParams, HeatmapRunner
from src.experiments.runners.info_flow import InfoFlowParams, InfoFlowRunner


def test_experiments_configs():
    variant_params = BaseVariantParams(
        model_arch=MODEL_ARCH.MAMBA1,
        model_size=TModelSize("130M"),
    )
    prompt_filteration = AllPromptFilteration(dataset_name=DatasetName.counter_fact)
    window_size = TWindowSize(10)
    metadata_params = MetadataParams(
        code_version=TCodeVersionName("test"),
    )
    evaluate_model_config = EvaluateModelRunner(
        variant_params=EvaluateModelParams(
            **asdict(variant_params),
        ),
        input_params=InputParams(filteration=prompt_filteration),
        metadata_params=metadata_params,
    )
    heatmap_config = HeatmapRunner(
        variant_params=HeatmapParams(
            **asdict(variant_params),
            window_size=window_size,
        ),
        input_params=InputParams(filteration=prompt_filteration),
        metadata_params=metadata_params,
    )
    info_flow_config = InfoFlowRunner(
        variant_params=InfoFlowParams(
            **asdict(variant_params),
            window_size=window_size,
            source=TokenType.last,
            target=TokenType.last,
            feature_category=FeatureCategory.ALL,
        ),
        input_params=InputParams(filteration=prompt_filteration),
        metadata_params=metadata_params,
    )
    assert evaluate_model_config
    assert heatmap_config
    assert info_flow_config
