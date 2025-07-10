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
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Константы ---

# Путь к датасету по умолчанию
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset.csv')

# --- Основной класс ---

class FPbase:

    '''
    Класс для представления базы данных FPbase для обучения модели предсказывать свойства
    Реализован как Singleton - может существовать только один экземпляр класса.

    Параметры:
        dataset_path (str): путь к .csv таблице FPbase (например, data/dataset.csv)
    
    Примеры:
        >>> from fpgen.prop_prediction.dataset import FPbase
        >>> dataset = FPbase('dataset.csv')
        >>> x_train, y_train = dataset.get_train('ex_max', is_scaled=False)
        >>> x_test, y_test = dataset.get_test('ex_max', is_scaled=True)
    '''

    _instance = None

    def __new__(cls, dataset_path: str | None = None):
        # Проврека на то, сущестует ли уже экземпляр этого класса
        if cls._instance is None:
            cls._instance = super(FPbase, cls).__new__(cls)
            cls._instance.__initialized = False
            
            # Если путь не указан, используем путь по умолчанию
            if dataset_path is None:
                dataset_path = DEFAULT_DATASET_PATH
                
            # Инициализируем экземпляр
            cls._instance.__init__(dataset_path)
        return cls._instance

    def __init__(self, dataset_path: str):
        if self.__initialized:
            return
        self.__initialized = True

        # Чтение данных
        self.__dataset: pd.DataFrame = pd.read_csv(dataset_path)

        self.properties: list = list(self.__dataset.columns)[1:]
        self.feature: str = self.__dataset.columns[0]
        
        # Разбиение данных
        self.__df_train, self.__df_test = train_test_split(
            self.__dataset, test_size=0.2, random_state=52
        )

        self.__scalers: Dict[str, StandardScaler] = {}

        # Добавление скейлеров для каждого свойства
        for property in self.properties:
            # Проверка: является ли колонка числовой. Если нет, то пропускаем
            if not pd.api.types.is_float_dtype(self.__df_train[property]):
                continue

            scaler = StandardScaler()
            scaler.fit(self.__df_train[[property]].dropna())
            self.__scalers[property] = scaler

    def get_train(self, target_name: str, is_scaled: bool = True) -> pd.DataFrame:

        '''
        Возвращает датафрейм (x, y) для переданного свойства из тренировочной выборки
        Параметр is_scaled отвечает за масштабирование таргетов
        '''

        not_nan_dataset = self.__df_train[[self.feature, target_name]].dropna()

        if is_scaled:
            not_nan_dataset[target_name] = self.__scalers[target_name].transform(
                not_nan_dataset[[target_name]]
            )

        return not_nan_dataset[self.feature], not_nan_dataset[target_name]

    def get_test(self, target_name: str, is_scaled: bool = True) -> pd.DataFrame:

        '''
        Возвращает датафрейм (x, y) для переданного свойства из тестовой выборки
        Параметр is_scaled отвечает за масштабирование таргетов
        '''

        not_nan_dataset = self.__df_test[[self.feature, target_name]].dropna()

        if is_scaled:
            not_nan_dataset[target_name] = self.__scalers[target_name].transform(
                not_nan_dataset[[target_name]]
            )

        return not_nan_dataset[self.feature], not_nan_dataset[target_name]

    def scale_targets(self, targets, target_name: str) -> np.ndarray:
        '''Масштабирует таргеты'''
        return self.__scalers[target_name].transform(pd.DataFrame(targets))
    
    def rescale_targets(self, targets, target_name: str) -> np.ndarray:
        '''Размасштабирует таргеты'''
        return self.__scalers[target_name].inverse_transform(pd.DataFrame(targets))