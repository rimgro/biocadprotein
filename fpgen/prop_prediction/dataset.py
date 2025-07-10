# =============================================================================
# prop_prediction/dataset.py
# =============================================================================
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

from typing import Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Основной класс ---

class FPbase:

    '''
    Класс для представления базы данных FPbase для обучения модели предсказывать свойства
    '''

    def __init__(self, dataset_path: str):
        self.__dataset: pd.DataFrame = pd.read_csv(dataset_path)

        self.properties: list = list(self.__dataset.columns)[1:]
        self.feature: str = self.__dataset.columns[0]
        
        self.__df_train, self.__df_test = train_test_split(
            self.__dataset, test_size=0.2, random_state=52
        )

        self.__scalers: Dict[str, StandardScaler] = {}

        for property in self.properties:
            # Проверка: является ли колонка числовой. Если нет, то пропускаем
            if not pd.api.types.is_float_dtype(self.__df_train[property]):
                continue

            scaler = StandardScaler()
            scaler.fit(self.__df_train[[property]].dropna())
            self.__scalers[property] = scaler

    def get_train(self, target_name: str, is_scaled: bool = True) -> pd.DataFrame:
        not_nan_dataset = self.__df_train[[self.feature, target_name]].dropna()

        if is_scaled:
            not_nan_dataset[target_name] = self.__scalers[target_name].transform(
                not_nan_dataset[[target_name]]
            )

        return not_nan_dataset[self.feature], not_nan_dataset[target_name]

    def get_test(self, target_name: str, is_scaled: bool = True) -> pd.DataFrame:
        not_nan_dataset = self.__df_test[[self.feature, target_name]].dropna()

        if is_scaled:
            not_nan_dataset[target_name] = self.__scalers[target_name].transform(
                not_nan_dataset[[target_name]]
            )

        return not_nan_dataset[self.feature], not_nan_dataset[target_name]

    def scale_targets(self, targets, target_name: str) -> np.ndarray:
        return self.__scalers[target_name].transform(pd.DataFrame(targets))
    
    def rescale_targets(self, targets, target_name: str) -> np.ndarray:
        return self.__scalers[target_name].inverse_transform(pd.DataFrame(targets))