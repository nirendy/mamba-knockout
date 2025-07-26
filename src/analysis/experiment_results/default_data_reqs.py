from src.analysis.prompt_filterations import (
    Correctness,
    get_shared_models_correctness_prompt_filteration,
)
from src.core.consts import (
    ALL_IMPORTANT_MODELS,
    GRAPHS_ORDER,
    TokenType,
)
from src.core.types import (
    FeatureCategory,
    TCodeVersionName,
    TWindowSize,
)
from src.data_ingestion.data_defs.data_defs import DataReqiermentCollection
from src.experiments.infrastructure.base_prompt_filteration import LogicalPromptFilteration, SamplePromptFilteration
from src.experiments.runners.heatmap import HeatmapParams
from src.experiments.runners.info_flow import InfoFlowParams

STANDARD_WINDOW_SIZE_FOR_INFO_FLOW = TWindowSize(9)
STANDARD_WINDOW_SIZE_FOR_HEATMAP = TWindowSize(5)
HEATMAP_WINDOW_SIZE = [TWindowSize(size) for size in [5, 9]]
ALL_WINDOW_SIZES = [TWindowSize(size) for size in [1, 3, 5, 9, 12, 15]]
MODEL_CORRECT_MODEL_CODE_VERSION = TCodeVersionName("v1")


def get_default_data_reqs() -> DataReqiermentCollection:
    data_reqs = DataReqiermentCollection()
    all_correct_prompt_filteration = get_shared_models_correctness_prompt_filteration(
        model_arch_and_sizes=list(ALL_IMPORTANT_MODELS.keys()),
        code_version=MODEL_CORRECT_MODEL_CODE_VERSION,
        correctness=Correctness.top_3_correct,
    )

    default_prompt_filteration = LogicalPromptFilteration.create_or(
        [
            # AllPromptFilteration(DATASETS.COUNTER_FACT, split=(SPLIT.TRAIN1,)),
            all_correct_prompt_filteration,
        ]
    )

    for model_arch_and_size in GRAPHS_ORDER:
        for target, source in [
            (TokenType.last, TokenType.last),
            (TokenType.last, TokenType.first),
            (TokenType.last, TokenType.subject),
            (TokenType.last, TokenType.relation),
            (TokenType.last, TokenType.relation_minus_last),
        ]:
            fcs = [FeatureCategory.ALL]
            if source in [
                TokenType.subject,
                TokenType.relation_minus_last,
                TokenType.relation,
            ]:
                fcs.extend([FeatureCategory.SLOW_DECAY, FeatureCategory.FAST_DECAY])
            for fc in fcs:
                data_reqs.add_data_req(
                    InfoFlowParams(
                        model_arch=model_arch_and_size.arch,
                        model_size=model_arch_and_size.size,
                        window_size=STANDARD_WINDOW_SIZE_FOR_INFO_FLOW,
                        source=source,
                        feature_category=fc,
                        target=target,
                    ),
                    default_prompt_filteration,
                )
        data_reqs.add_data_req(
            HeatmapParams(
                model_arch=model_arch_and_size.arch,
                model_size=model_arch_and_size.size,
                window_size=STANDARD_WINDOW_SIZE_FOR_HEATMAP,
            ),
            SamplePromptFilteration(
                base_prompt_filteration=default_prompt_filteration,
                sample_size=100,
                seed=42,
            ),
        )

    return data_reqs
