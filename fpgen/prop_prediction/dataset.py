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
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'fpbase.csv')
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
            feature_column: str = 'sequence',
            random_state: int = 52
        ) -> None:
        if dataset_path is None:
            dataset_path = DEFAULT_DATASET_PATH

        # Чтение данных
        self.__dataset: pd.DataFrame = pd.read_csv(dataset_path)

        # Публичные поля
        self.targets: list = list(self.__dataset.drop(columns=[feature_column]).columns)
        self.feature: str = feature_column

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

    def __preprocess_dataframe(
            self,
            dataframe: pd.DataFrame,
            target_name: str,
            is_scaled: bool = True
        ) -> Tuple[pd.Series, pd.Series]:

        '''
        Обрабатывает датафрейм и возвращает (x, y)

        Парметры:
            target_name (str): название таргета
            is_scaled (bool), опц.: нужно ли масштабировать таргеты
        '''

        # Обработанный датасет
        processed_dataset: pd.DataFrame = dataframe[[self.feature, target_name]].dropna() \
                                                                                .reset_index(drop=True)

        # Масштабирование признаков
        if is_scaled:
            processed_dataset[target_name] = self.__scalers[target_name].transform(
                processed_dataset[[target_name]]
            )

        # Получение пар (x, y)
        x = processed_dataset[self.feature]
        y = processed_dataset[target_name].values

        return x, y

    def get_train(self, target_name: str, is_scaled: bool = True) -> Tuple[pd.Series, pd.Series]:

        '''
        Возвращает пары (x, y) для переданного свойства из тренировочной выборки
        Параметры: см. FPbase.__preprocess_dataframe
        '''

        return self.__preprocess_dataframe(self.__df_train, target_name, is_scaled)

    def get_test(self, target_name: str, is_scaled: bool = True) -> Tuple[pd.Series, pd.Series]:

        '''
        Возвращает пары (x, y) для переданного свойства из тестовой выборки
        Парметры: см. FPbase.__preprocess_dataframe
        '''

        return self.__preprocess_dataframe(self.__df_test, target_name, is_scaled)

    def scale_targets(self, targets: np.ndarray | pd.Series, target_name: str) -> np.ndarray | pd.Series:

        '''
        Масштабирует таргеты
        
        Параметры:
            targets (np.ndarray | pd.Series): список таргетов
            target_name (str): название таргета, который нужно масштабировать
        '''

        if not self.is_regression_target(target_name):
            return targets
        
        return self.__scalers[target_name].transform(pd.DataFrame(targets))
    
    def rescale_targets(self, targets: np.ndarray | pd.Series, target_name: str) -> np.ndarray | pd.Series:

        '''
        Размасштабирует таргеты
        Параметры: см. FPbase.scale_targets
        '''

        if not self.is_regression_target(target_name):
            return targets
        
        return self.__scalers[target_name].inverse_transform(pd.DataFrame(targets))
    
    def to_dataframe(self) -> pd.DataFrame:

        '''
        Преобразовывает датасет в датафрейм

        Возвращает:
            pd.Dataframe
        '''

        return self.__dataset
    
    def to_train_dataframe(self, is_scaled: bool = True) -> pd.DataFrame:

        '''
        Возвращает тренировочный датафрейм
        Параметр is_scaled отвечает за масштабирование таргетов
        '''

        # Обработанный датафрейм
        processed_df = self.__df_train.copy().reset_index(drop=True)
        
        # Масштабирование таргетов если нужно
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
        
        # Обработанный датафрейм
        processed_df = self.__df_test.copy().reset_index(drop=True)

        # Масштабирование таргетов если нужно
        if is_scaled:
            for target in self.targets:
                if self.is_regression_target(target):
                    processed_df[target] = self.scale_targets(processed_df[target], target)

        return processed_df
    
    def is_regression_target(self, target_name: str) -> bool:

        '''
        Проверяем, является ли свойство регрессионным
        '''

        return target_name in self.regression_targets
    
    def is_classification_target(self, target_name: str) -> bool:

        '''
        Проверяем, является ли свойство классифицируемым
        '''

        return target_name in self.classification_targets