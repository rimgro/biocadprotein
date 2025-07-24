# =============================================================================
# prop_prediction/predictor.py
# =============================================================================
# Главный интерфейс предсказания свойств белков.
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

from fpgen.prop_prediction.pipeline  import PredictionPipeline
from fpgen.prop_prediction.registry  import get_pipeline

class PropertiesPredictor:
    def __init__(self, model_name: str):
        self.pipeline: PredictionPipeline = get_pipeline(model_name)

    def predict(self, sequences: list[str] | str, target: str):
        if isinstance(sequences, str):
            sequences = [sequences]
        return self.pipeline.run(sequences, target)