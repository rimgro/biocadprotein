# =============================================================================
# api.py
# =============================================================================
# Интерфейс для работы с библиотекой
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

from fpgen.generation.generator import ProteinGenerator
from fpgen.prop_prediction.predictor import PropertiesPredictor

__all__ = ['ProteinGenerator', 'PropertiesPredictor']
