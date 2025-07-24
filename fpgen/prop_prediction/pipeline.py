# =============================================================================
# prop_prediction/pipeline.py
# =============================================================================
# Абстрактная реализация пайплайна для моделей предсказания спектральных
# свойств флуоресцентных молекул
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

from abc import ABC, abstractmethod

# --- Основной класс ---

class PredictionPipeline(ABC):

    '''
    Абстрактный класс для пайплайна предсказания.
    '''

    @abstractmethod
    def preprocess(self, sequences: list[str]) -> any:
        pass

    @abstractmethod
    def predict(self, inputs) -> any:
        pass

    @abstractmethod
    def postprocess(self, raw_preds, target: str) -> any:
        pass

    def run(self, sequences: list[str], target: str):
        inputs = self.preprocess(sequences)
        raw_preds = self.predict(inputs)
        return self.postprocess(raw_preds, target)
