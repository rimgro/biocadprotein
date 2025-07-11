# =============================================================================
# prop_prediction/dataset.py
# =============================================================================
# Модуль для работы с датасетом для создания модели предсказания свойств белковых молекул
#
# Часть проекта с проектной смены "Большие Вызовы"
#
# Лицензия: MIT (см. LICENSE)
# =============================================================================

import os
from typing import Dict, Callable, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Константы ---

# Путь к датасету по умолчанию
DEFAULT_DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dataset.csv')
)

# --- Основной класс ---

class FPbase:

    '''
    Класс для представления базы данных FPbase для обучения модели предсказывать свойства

    Параметры:
        dataset_path (str): путь к .csv таблице FPbase (например, data/dataset.csv)
    
    Примеры:
        >>> from fpgen.prop_prediction.dataset import FPbase
        >>> dataset = FPbase('dataset.csv')
        >>> x_train, y_train = dataset.get_train('ex_max', is_scaled=False)
        >>> x_test, y_test = dataset.get_test('ex_max', is_scaled=True)
    '''

    def __init__(
            self,
            dataset_path: str | None = None,
            preprocess_function: Callable | None = None,
            random_state: int = 52
        ) -> None:
        if dataset_path is None:
            dataset_path = DEFAULT_DATASET_PATH

        # Чтение данных
        self.__dataset: pd.DataFrame = pd.read_csv(dataset_path)

        # Публичные поля
        self.targets: list = list(self.__dataset.columns)[1:]
        self.feature: str = self.__dataset.columns[0]

        self.regression_targets = []
        self.classification_targets = []

        # Если передана функция для предобработки, то предобрабатываем все X
        if preprocess_function is not None:
            self.__dataset[self.feature] = self.__dataset[self.feature].apply(preprocess_function)
        
        # Разбиение данных
        self.__df_train, self.__df_test = train_test_split(
            self.__dataset, test_size=0.2, random_state=random_state
        )

        self.__scalers: Dict[str, StandardScaler] = {}

        # Добавление скейлеров для каждого свойства
        for target in self.targets:
            # Проверка: является ли колонка числовой. Если нет, то пропускаем
            if not pd.api.types.is_float_dtype(self.__df_train[target]):
                self.classification_targets.append(target)
                continue

            self.regression_targets.append(target)

            # Обучение скейлера
            scaler: StandardScaler = StandardScaler()
            scaler.fit(self.__df_train[[target]].dropna())
            self.__scalers[target] = scaler

    def get_train(self, target_name: str, is_scaled: bool = True) -> Tuple[pd.Series, pd.Series]:

        '''
        Возвращает датафрейм (x, y) для переданного свойства из тренировочной выборки
        Параметр is_scaled отвечает за масштабирование таргетов
        '''

        not_nan_dataset: pd.DataFrame = self.__df_train[[self.feature, target_name]].dropna() \
                                                                                    .reset_index(drop=True)

        if is_scaled:
            not_nan_dataset[target_name] = self.__scalers[target_name].transform(
                not_nan_dataset[[target_name]]
            )

        return not_nan_dataset[self.feature], not_nan_dataset[target_name]

    def get_test(self, target_name: str, is_scaled: bool = True) -> Tuple[pd.Series, pd.Series]:

        '''
        Возвращает датафрейм (x, y) для переданного свойства из тестовой выборки
        Параметр is_scaled отвечает за масштабирование таргетов

        TODO: перенести get_train и get_test в одну функцию и вызывать ее из этих методов
        '''

        not_nan_dataset: pd.DataFrame = self.__df_test[[self.feature, target_name]].dropna() \
                                                                                   .reset_index(drop=True)

        if is_scaled:
            not_nan_dataset[target_name] = self.__scalers[target_name].transform(
                not_nan_dataset[[target_name]]
            )

        return not_nan_dataset[self.feature], not_nan_dataset[target_name]

    def scale_targets(self, targets: np.ndarray | pd.Series, target_name: str) -> np.ndarray | pd.Series:
        '''Масштабирует таргеты'''
        if not self.is_regression_target(target_name):
            return targets
        return self.__scalers[target_name].transform(pd.DataFrame(targets))
    
    def rescale_targets(self, targets: np.ndarray | pd.Series, target_name: str) -> np.ndarray | pd.Series:
        '''Размасштабирует таргеты'''
        if not self.is_regression_target(target_name):
            return targets
        return self.__scalers[target_name].inverse_transform(pd.DataFrame(targets))
    
    def to_dataframe(self) -> pd.DataFrame:
        return self.__dataset
    
    def to_train_dataframe(self, is_scaled: bool = True) -> pd.DataFrame:

        '''
        Возвращает тренировочный датафрейм
        Параметр is_scaled отвечает за масштабирование таргетов
        '''

        processed_df = self.__df_train.copy().reset_index(drop=True)

        if is_scaled:
            for target in self.targets:
                if self.is_regression_target(target):
                    processed_df[target] = self.scale_targets(processed_df[target], target)

        return processed_df
    
    def to_test_dataframe(self, is_scaled: bool = True) -> pd.DataFrame:

        '''
        Возвращает тестовый датафрейм
        Параметр is_scaled отвечает за масштабирование таргетов
        '''
        
        processed_df = self.__df_test.copy().reset_index(drop=True)

        if is_scaled:
            for target in self.targets:
                if self.is_regression_target(target):
                    processed_df[target] = self.scale_targets(processed_df[target], target)

        return processed_df
    
    def is_regression_target(self, target_name: str) -> bool:
        '''Проверяем, является ли свойство регрессионным'''
        return target_name in self.regression_targets
    
    def is_classification_target(self, target_name: str) -> bool:
        '''Проверяем, является ли свойство классифицируемым'''
        return target_name in self.classification_targets