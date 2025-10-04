import pandas as pd
import numpy as np
import random
import math
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from datetime import datetime, timedelta


class GasLevelModel:
    """
    Modelo para análisis y predicción de niveles de gas metano en sistemas embebidos
    de minas subterráneas, considerando factores operacionales como temperatura,
    humedad, tiempo de calibración y nivel de batería.
    """

    def __init__(self, model_path='./trained_models/regression_linear_model.pkl'):
        """
        Inicializa el modelo con valores por defecto y carga modelo existente si está disponible.
        """
        self.model = None  # Modelo de regresión lineal
        self.scaler = None  # Escalador para normalización de datos
        self.df = None  # DataFrame con los datos cargados
        self.X = None  # Variables predictoras
        self.y = None  # Variable objetivo (nivel_gas_metano)
        self.last_known_values = None  # Últimos valores conocidos para predicciones
        self.model_path = model_path  # Ruta para guardar/cargar el modelo
        self.training_results = None  # Almacena los resultados del entrenamiento
        self.feature_names = ['temperatura_sensor', 'humedad_ambiente',
                              'tiempo_desde_calibracion', 'nivel_bateria']
        self.correction_history = {}  # Historial de correcciones aplicadas

        # Intentar cargar modelo existente al inicializar
        self._load_trained_model()

    def _load_trained_model(self):
        """
        Carga un modelo previamente entrenado desde el archivo.

        Returns:
            bool: True si el modelo se cargó correctamente, False en caso contrario.
        """
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.training_results = saved_data.get('training_results')
                self.X = saved_data.get('X')
                self.y = saved_data.get('y')
                self.df = saved_data.get('df')
                print("Modelo de regresión lineal cargado exitosamente desde", self.model_path)
                return True
            except Exception as e:
                print(f"Error al cargar el modelo: {str(e)}")
                self.model = None
                self.scaler = None
                self.training_results = None
                self.X = None
                self.y = None
                self.df = None
        return False

    def _save_trained_model(self):
        """
        Guarda el modelo entrenado con el scaler y los resultados obtenidos del entrenamiento.

        Returns:
            bool: True si el modelo se guardó correctamente, False en caso contrario.
        """
        if self.model is not None and self.scaler is not None:
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                with open(self.model_path, 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'scaler': self.scaler,
                        'training_results': self.training_results,
                        'X': self.X,
                        'y': self.y,
                        'df': self.df
                    }, f)
                print("Modelo de regresión lineal guardado exitosamente en", self.model_path)
                return True
            except Exception as e:
                print(f"Error al guardar el modelo: {str(e)}")
        return False

    def get_training_results(self):
        """
        Retorna los resultados del entrenamiento sin reentrenar.

        Returns:
            dict: Resultados del entrenamiento
        """
        if self.training_results is None:
            if self.model is None:
                raise ValueError("El modelo no está entrenado. Ejecute train_model() primero.")
            # Si no hay resultados guardados pero el modelo existe, los recalculamos una vez
            self._calculate_training_results()
        return self.training_results

    def _calculate_training_results(self):
        """
        Calcula las métricas de rendimiento del modelo entrenado.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("El modelo no está entrenado.")

        # Normalizar variables predictoras
        X_scaled = self.scaler.transform(self.X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.X.columns)

        # Dividir datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42
        )

        # Realizar predicciones en conjunto de prueba
        y_pred = self.model.predict(X_test)

        # Conversión de métricas
        metric_y_test = self.ppm_to_percentage(y_test)
        metric_y_pred = self.ppm_to_percentage(y_pred)

        # Conversión de coeficientes
        feature_importance_coefficients = self.model.coef_.tolist()

        # Conversión de residuales
        residuals_values = (y_test - y_pred).tolist()
        residuals_predictions = y_pred.tolist()

        # Análisis de normalidad de residuos
        _, p_value = stats.shapiro(y_test - y_pred)

        self.training_results = {
            'metrics': {
                'mse': float(mean_squared_error(metric_y_test, metric_y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(metric_y_test, metric_y_pred))),
                'mae': float(mean_absolute_error(metric_y_test, metric_y_pred)),
                'r2': float(r2_score(metric_y_test, metric_y_pred))
            },
            'prediction_data': {
                'real_values': self.ppm_to_percentage(y_test.tolist()),
                'predicted_values': self.ppm_to_percentage(y_pred.tolist())
            },
            'feature_importance': {
                'variables': self.X.columns.tolist(),
                'coefficients': self.ppm_to_percentage(feature_importance_coefficients)
            },
            'residuals': {
                'values': self.ppm_to_percentage(residuals_values),
                'predictions': self.ppm_to_percentage(residuals_predictions)
            },
            'residual_normality': {
                'shapiro_p_value': float(p_value),
                'is_normal': p_value > 0.05,
                'interpretation': "Los residuos siguen una distribución normal" if p_value > 0.05 else
                "Los residuos no siguen una distribución normal"
            }
        }

    def load_data(self, csv_path):
        """
        Carga los datos desde un archivo CSV y realiza preprocesamiento básico.

        Args:
            csv_path (str): Ruta al archivo CSV con los datos del sensor

        Returns:
            dict: Estadísticas descriptivas básicas del conjunto de datos
        """
        # Carga de datos
        self.df = pd.read_csv(csv_path)

        # Procesamiento de campos temporales
        self.df['datetime'] = pd.to_datetime(self.df['fecha'] + ' ' + self.df['hora'])
        self.df['dias_desde_calibracion'] = self.df['tiempo_desde_calibracion'] / 24

        # Definir variables predictoras (X) y variable objetivo (y)
        self.X = self.df[self.feature_names]
        self.y = self.df['nivel_gas_metano']

        # Verificación de datos faltantes
        missing_data = self.df.isnull().sum()

        # Actualización de últimos valores conocidos para predicciones
        self.update_last_known_values()

        return {
            "descriptive_stats": self.df.describe().replace({np.nan: None}).to_dict(),
            'data_shape': self.df.shape,
            'missing_data': missing_data.to_dict()
        }

    def get_basic_stats(self):
        """
        Retorna estadísticas descriptivas del conjunto de datos.

        Returns:
            dict: Estadísticas descriptivas y dimensiones del dataset
        """
        if self.df is None:
            return {"error": "No data loaded. Call load_data() first."}

        return {
            "descriptive_stats": self.df.describe().replace({np.nan: None}).to_dict(),
            'data_shape': self.df.shape
        }

    def detect_outliers(self, columns, threshold=3):
        """
        Detecta valores atípicos en las columnas especificadas utilizando Z-scores.

        Args:
            columns (list): Lista de columnas a analizar
            threshold (float): Umbral Z-score para identificar outliers

        Returns:
            dict: Información sobre los valores atípicos detectados
        """
        if self.df is None:
            return {"error": "No data loaded. Call load_data() first."}

        outliers = {}
        for column in columns:
            # Cálculo de Z-scores (número de desviaciones estándar desde la media)
            z_scores = np.abs(stats.zscore(self.df[column]))

            # Identificación de índices con Z-scores mayores al umbral
            outlier_indices = np.where(z_scores > threshold)[0]

            # Almacenamiento de información sobre outliers
            outliers[column] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(self.df)) * 100,
                'values': self.df.loc[outlier_indices, column].tolist(),
                'indices': outlier_indices.tolist(),
                'statistics': {
                    'mean': self.df[column].mean(),
                    'std': self.df[column].std(),
                    'min': self.df[column].min(),
                    'max': self.df[column].max()
                }
            }
        return {
            "outliers": outliers
        }

    def correct_outliers(self, columns, threshold=3, method='mean'):
        """
        Corrige valores atípicos en las columnas especificadas.

        Args:
            columns (list): Lista de columnas a analizar
            threshold (float): Umbral Z-score para identificar outliers
            method (str): Método de corrección ('mean', 'median', 'trim', 'winsorize')

        Returns:
            dict: Estadísticas sobre correcciones realizadas
        """
        if self.df is None:
            return {"error": "No data loaded. Call load_data() first."}

        corrections = {}

        for column in columns:
            # Cálculo de Z-scores para detectar outliers
            z_scores = np.abs(stats.zscore(self.df[column]))
            outlier_indices = np.where(z_scores > threshold)[0]

            if len(outlier_indices) > 0:
                # Guardar valores originales antes de la corrección
                original_values = self.df.loc[outlier_indices, column].copy()

                if method == 'mean':
                    # Reemplazar con la media de la columna
                    replacement = self.df[column].mean()
                    self.df.loc[outlier_indices, column] = replacement

                elif method == 'median':
                    # Reemplazar con la mediana (más robusta ante outliers)
                    replacement = self.df[column].median()
                    self.df.loc[outlier_indices, column] = replacement

                elif method == 'trim':
                    # Recortar valores extremos usando IQR (rango intercuartil)
                    q1, q3 = np.percentile(self.df[column], [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    for idx in outlier_indices:
                        if self.df.loc[idx, column] < lower_bound:
                            self.df.loc[idx, column] = lower_bound
                        elif self.df.loc[idx, column] > upper_bound:
                            self.df.loc[idx, column] = upper_bound

                elif method == 'winsorize':
                    # Winsorización: limita valores extremos a percentiles
                    lower_bound, upper_bound = np.percentile(self.df[column], [5, 95])
                    for idx in outlier_indices:
                        if self.df.loc[idx, column] < lower_bound:
                            self.df.loc[idx, column] = lower_bound
                        elif self.df.loc[idx, column] > upper_bound:
                            self.df.loc[idx, column] = upper_bound

                # Registrar correcciones realizadas
                corrections[column] = {
                    'count': len(outlier_indices),
                    'percentage': (len(outlier_indices) / len(self.df)) * 100,
                    'original_values': original_values.tolist(),
                    'method': method,
                    'indices': outlier_indices.tolist()
                }

                # Si se usó un valor específico para el reemplazo, guardarlo
                if method in ['mean', 'median']:
                    corrections[column]['replacement_value'] = replacement

        # Guardar historial de correcciones
        self.correction_history = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'corrections': corrections
        }

        return {"corrections": corrections}

    def impute_missing_values(self, columns, method='mean'):
        """
        Imputa valores faltantes en las columnas especificadas.

        Args:
            columns (list): Lista de columnas a procesar
            method (str): Método de imputación ('mean', 'median', 'ffill', 'bfill')

        Returns:
            dict: Estadísticas sobre valores imputados
        """
        if self.df is None:
            return {"error": "No data loaded. Call load_data() first."}

        imputation_stats = {}

        for column in columns:
            # Contar valores faltantes iniciales
            missing_count = self.df[column].isnull().sum()

            if missing_count > 0:
                if method == 'mean':
                    # Imputar con la media
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                elif method == 'median':
                    # Imputar con la mediana
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                elif method == 'ffill':
                    # Propagar último valor válido hacia adelante
                    self.df[column].fillna(method='ffill', inplace=True)
                elif method == 'bfill':
                    # Propagar próximo valor válido hacia atrás
                    self.df[column].fillna(method='bfill', inplace=True)

                # Verificar valores aún faltantes
                remaining_missing = self.df[column].isnull().sum()

                imputation_stats[column] = {
                    'initial_missing': int(missing_count),
                    'remaining_missing': int(remaining_missing),
                    'imputed_count': int(missing_count - remaining_missing),
                    'method': method
                }

        return {"imputations": imputation_stats}

    def analyze_temporal_degradation(self):
        """
        Analiza la degradación temporal del sensor en relación al tiempo desde calibración.

        Returns:
            dict: Datos para análisis de tendencia y degradación
        """
        if self.df is None:
            return {"error": "No data loaded. Call load_data() first."}

        # Agrupar por días desde calibración y calcular estadísticas
        error_by_day = self.df.groupby('dias_desde_calibracion').agg({
            'nivel_gas_metano': ['mean', 'std', 'count']
        }).reset_index()

        # Calcular tendencia lineal
        days = error_by_day['dias_desde_calibracion'].values
        gas_levels = error_by_day['nivel_gas_metano']['mean'].values

        if len(days) > 1:
            # Ajuste lineal para estimar tasa de degradación
            slope, intercept = np.polyfit(days, gas_levels, 1)
            trend_line = intercept + slope * days

            degradation_analysis = {
                'trend_slope': float(slope),
                'trend_intercept': float(intercept),
                'degradation_rate_per_day': float(slope),
                'days': days.tolist(),
                'trend_values': trend_line.tolist()
            }
        else:
            degradation_analysis = {
                'error': 'Insufficient data for trend analysis'
            }

        return {
            'days': error_by_day['dias_desde_calibracion'].tolist(),
            'std_dev': error_by_day['nivel_gas_metano']['std'].replace({np.nan: None}).tolist(),
            'mean': error_by_day['nivel_gas_metano']['mean'].tolist(),
            'count': error_by_day['nivel_gas_metano']['count'].tolist(),
            'degradation_analysis': degradation_analysis
        }

    def analyze_battery_impact(self):
        """
        Analiza el impacto del nivel de batería en las mediciones de gas.

        Returns:
            dict: Análisis estadístico del impacto del nivel de batería
        """
        if self.df is None:
            return {"error": "No data loaded. Call load_data() first."}

        # Agrupar por rangos de nivel de batería
        battery_ranges = [0, 20, 40, 60, 80, 100]
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

        self.df['battery_range'] = pd.cut(
            self.df['nivel_bateria'],
            bins=battery_ranges,
            labels=labels,
            include_lowest=True
        )

        # Calcular estadísticas por grupo de batería
        battery_analysis = self.df.groupby('battery_range').agg({
            'nivel_gas_metano': ['mean', 'std', 'count'],
            'temperatura_sensor': ['mean'],
            'humedad_ambiente': ['mean']
        })

        # Convertir a formato más simple
        result = {}
        for battery_range in battery_analysis.index:
            if battery_range is not None:  # Evitar problemas con NaN

                gas_mean_ppm = float(battery_analysis.loc[battery_range, ('nivel_gas_metano', 'mean')])
                gas_std_ppm = float(battery_analysis.loc[battery_range, ('nivel_gas_metano', 'std')])

                result[str(battery_range)] = {
                    'gas_mean': self.ppm_to_percentage(gas_mean_ppm),
                    'gas_std': self.ppm_to_percentage(gas_std_ppm),
                    'count': int(battery_analysis.loc[battery_range, ('nivel_gas_metano', 'count')]),
                    'temp_mean': float(battery_analysis.loc[battery_range, ('temperatura_sensor', 'mean')]),
                    'humidity_mean': float(battery_analysis.loc[battery_range, ('humedad_ambiente', 'mean')])
                }

        # Calcular correlación entre nivel de batería y nivel de gas
        correlation = self.df[['nivel_bateria', 'nivel_gas_metano']].corr().loc['nivel_bateria', 'nivel_gas_metano']

        # Eliminar la columna temporal creada para este análisis
        self.df.drop('battery_range', axis=1, inplace=True)

        return {
            'battery_ranges': result,
            'correlation': float(correlation),
            'interpretation': self._interpret_correlation(correlation)
        }

    def analyze_temperature_impact(self):
        """
        Analiza el impacto de la temperatura del sensor en las mediciones de gas.

        Returns:
            dict: Análisis estadístico del impacto de la temperatura
        """
        if self.df is None:
            return {"error": "No data loaded. Call load_data() first."}

        # Agrupar por rangos de temperatura
        temp_min = self.df['temperatura_sensor'].min()
        temp_max = self.df['temperatura_sensor'].max()

        # Crear 5 rangos equidistantes
        temp_step = (temp_max - temp_min) / 5
        temp_ranges = [temp_min + i * temp_step for i in range(6)]

        self.df['temp_range'] = pd.cut(
            self.df['temperatura_sensor'],
            bins=temp_ranges,
            include_lowest=True
        )

        # Calcular estadísticas por grupo de temperatura
        temp_analysis = self.df.groupby('temp_range').agg({
            'nivel_gas_metano': ['mean', 'std', 'count'],
            'nivel_bateria': ['mean'],
            'humedad_ambiente': ['mean']
        })

        # Convertir a formato más simple
        result = {}
        for temp_range in temp_analysis.index:
            if temp_range is not None:
                range_str = f"{temp_range.left:.1f}-{temp_range.right:.1f}°C"
                gas_mean_ppm = float(temp_analysis.loc[temp_range, ('nivel_gas_metano', 'mean')])
                gas_std_ppm = float(temp_analysis.loc[temp_range, ('nivel_gas_metano', 'std')])
                result[range_str] = {
                    'gas_mean': self.ppm_to_percentage(gas_mean_ppm),
                    'gas_std': self.ppm_to_percentage(gas_std_ppm),
                    'count': int(temp_analysis.loc[temp_range, ('nivel_gas_metano', 'count')]),
                    'battery_mean': float(temp_analysis.loc[temp_range, ('nivel_bateria', 'mean')]),
                    'humidity_mean': float(temp_analysis.loc[temp_range, ('humedad_ambiente', 'mean')])
                }

        # Calcular correlación entre temperatura y nivel de gas
        correlation = self.df[['temperatura_sensor', 'nivel_gas_metano']].corr().loc[
            'temperatura_sensor', 'nivel_gas_metano']

        # Eliminar la columna temporal creada para este análisis
        self.df.drop('temp_range', axis=1, inplace=True)

        return {
            'temperature_ranges': result,
            'correlation': float(correlation),
            'interpretation': self._interpret_correlation(correlation)
        }

    def get_correlations(self):
        """
        Calcula la matriz de correlación entre las variables clave.

        Returns:
            dict: Matriz de correlaciones entre variables
        """
        if self.df is None:
            return {"error": "No data loaded. Call load_data() first."}

        columns = [
            'temperatura_sensor',
            'humedad_ambiente',
            'tiempo_desde_calibracion',
            'nivel_gas_metano',
            'nivel_bateria'
        ]

        correlations = self.df[columns].corr().to_dict()

        # Añadir interpretaciones para correlaciones principales
        interpretations = {}

        for var in ['temperatura_sensor', 'humedad_ambiente', 'tiempo_desde_calibracion', 'nivel_bateria']:
            corr_value = self.df[['nivel_gas_metano', var]].corr().loc['nivel_gas_metano', var]
            interpretations[f"{var}_gas"] = self._interpret_correlation(corr_value)

        return {
            "correlations": correlations,
            "interpretations": interpretations
        }

    def _interpret_correlation(self, correlation_value):
        """
        Interpreta el valor de correlación en términos descriptivos.

        Args:
            correlation_value (float): Valor de correlación (-1 a 1)

        Returns:
            str: Interpretación descriptiva
        """
        abs_corr = abs(correlation_value)

        if abs_corr < 0.1:
            strength = "insignificante"
        elif abs_corr < 0.3:
            strength = "débil"
        elif abs_corr < 0.5:
            strength = "moderada"
        elif abs_corr < 0.7:
            strength = "fuerte"
        else:
            strength = "muy fuerte"

        direction = "positiva" if correlation_value > 0 else "negativa"

        if abs_corr < 0.1:
            return f"Correlación {strength} (prácticamente no hay relación lineal)"
        else:
            return f"Correlación {strength} {direction} ({correlation_value:.3f})"

    def train_model(self, force_retrain=False):
        """
        Entrena el modelo de regresión lineal para predecir niveles de gas, con
        validación de modelo existente y opción de forzar reentrenamiento.

        Args:
            force_retrain (bool): Si es True, fuerza el reentrenamiento incluso si ya existe

        Returns:
            dict: Métricas de evaluación y datos de predicción
        """
        # Validar si ya existe un modelo entrenado
        if self.model is not None and not force_retrain:
            return self.get_training_results()

        if self.df is None:
            return {"error": "No data loaded. Call load_data() first."}

        columns_to_process = [
            'temperatura_sensor',
            'humedad_ambiente',
            'nivel_gas_metano',
            'nivel_bateria'
        ]

        # Detectar outliers
        outliers_info = self.detect_outliers(columns_to_process)

        # Imputar valores faltantes
        imputations_info = self.impute_missing_values(
            columns=columns_to_process,
            method='median'  # La mediana suele ser robusta para datos asimétricos
        )

        # Corregir datos atípicos
        corrections_info = self.correct_outliers(
            columns=columns_to_process,
            threshold=3,
            method='median'  # La mediana es generalmente más robusta para outliers
        )

        # Definir variables predictoras (X) y variable objetivo (y)
        self.X = self.df[self.feature_names]
        self.y = self.df['nivel_gas_metano']

        # Normalizar variables predictoras
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.X.columns)

        # Dividir datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42
        )

        # Entrenar modelo de regresión lineal
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Calcular y guardar resultados del entrenamiento
        self._calculate_training_results()

        # Guardar el modelo entrenado
        self._save_trained_model()

        # Añadir información de preprocesamiento a los resultados
        self.training_results['preprocessing'] = {
            'outliers': outliers_info,
            'imputations': imputations_info,
            'corrections': corrections_info
        }

        return self.training_results

    def predict(self, temperatura, humedad, tiempo_calibracion, nivel_bateria):
        """
        Realiza una predicción puntual del nivel de gas con análisis de confiabilidad.

        Args:
            temperatura (float): Temperatura del sensor
            humedad (float): Humedad ambiente
            tiempo_calibracion (float): Tiempo desde la última calibración (horas)
            nivel_bateria (float): Nivel de batería (%)

        Returns:
            dict: Predicción de nivel de gas y análisis de confiabilidad
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado o cargado correctamente.")

        # Preparar datos para predicción
        nuevos_datos = pd.DataFrame(
            [[temperatura, humedad, tiempo_calibracion, nivel_bateria]],
            columns=self.X.columns
        )

        # Normalizar datos
        nuevos_datos_scaled = pd.DataFrame(
            self.scaler.transform(nuevos_datos),
            columns=self.X.columns
        )

        # Realizar predicción
        prediction = float(self.model.predict(nuevos_datos_scaled)[0])

        # Añadir análisis de confiabilidad
        reliability_analysis = self._assess_prediction_reliability(
            temperatura, humedad, tiempo_calibracion, nivel_bateria
        )

        return {
            'predicted_gas_level': self.ppm_to_percentage(prediction),
            'reliability_analysis': reliability_analysis
        }

    def _assess_prediction_reliability(self, temperatura, humedad, tiempo_calibracion, nivel_bateria):
        """
        Evalúa la confiabilidad de una predicción basada en los valores de entrada.

        Args:
            temperatura (float): Temperatura del sensor
            humedad (float): Humedad ambiente
            tiempo_calibracion (float): Tiempo desde la última calibración (horas)
            nivel_bateria (float): Nivel de batería (%)

        Returns:
            dict: Análisis de confiabilidad
        """
        reliability_factors = {}
        overall_reliability = 100.0  # Comenzamos con 100% de confiabilidad

        # Verificar si la temperatura está dentro del rango normal
        temp_min, temp_max = self.df['temperatura_sensor'].min(), self.df['temperatura_sensor'].max()
        if temperatura < temp_min or temperatura > temp_max:
            reliability_factors['temperatura'] = {
                'factor': 'Temperatura fuera de rango de entrenamiento',
                'impact': 'alto',
                'reduction': 30
            }
            overall_reliability -= 30

        # Verificar nivel de batería
        if nivel_bateria < 20:
            reliability_factors['bateria'] = {
                'factor': 'Nivel de batería muy bajo',
                'impact': 'alto',
                'reduction': 40
            }
            overall_reliability -= 40
        elif nivel_bateria < 50:
            reliability_factors['bateria'] = {
                'factor': 'Nivel de batería moderadamente bajo',
                'impact': 'medio',
                'reduction': 15
            }
            overall_reliability -= 15

        # Verificar tiempo desde calibración
        calibration_max = self.df['tiempo_desde_calibracion'].max()
        if tiempo_calibracion > calibration_max:
            reliability_factors['calibracion'] = {
                'factor': 'Tiempo desde calibración excede datos de entrenamiento',
                'impact': 'medio',
                'reduction': 20
            }
            overall_reliability -= 20
        elif tiempo_calibracion > (calibration_max * 0.8):
            reliability_factors['calibracion'] = {
                'factor': 'Tiempo desde calibración cercano al límite de entrenamiento',
                'impact': 'bajo',
                'reduction': 10
            }
            overall_reliability -= 10

        # Asegurar que la confiabilidad no sea negativa
        overall_reliability = max(0, overall_reliability)

        return {
            'overall_reliability_percentage': overall_reliability,
            'reliability_factors': reliability_factors,
            'reliability_level': 'alta' if overall_reliability >= 80 else
            'media' if overall_reliability >= 50 else 'baja'
        }

    def update_last_known_values(self):
        """
        Actualiza los últimos valores conocidos para predicciones futuras.
        """
        if self.df is not None and not self.df.empty:
            latest_row = self.df.iloc[-1]
            current_time = datetime.now()

            # Aplicar variaciones dentro de rangos realistas
            temp_current = latest_row['temperatura_sensor']
            # Variación de temperatura (±2°C) manteniendo límites 0-49
            temp_variation = max(0, min(49, temp_current + random.uniform(-2, 2)))

            hum_current = latest_row['humedad_ambiente']
            # Variación de humedad (±3%) manteniendo límites 61-95
            hum_variation = max(61, min(95, hum_current + random.uniform(-3, 3)))

            # Nivel de batería con degradación realista (0.1-0.3% por hora)
            battery_drain = random.uniform(0.1, 0.3)
            new_battery = max(10, min(100, latest_row['nivel_bateria'] - battery_drain))

            # Tiempo desde calibración (aumenta con el tiempo real)
            hours_passed = (current_time - pd.to_datetime(latest_row['datetime'])).total_seconds() / 3600
            new_calibration_time = min(719, latest_row['tiempo_desde_calibracion'] + hours_passed)

            self.last_known_values = {
                'datetime': current_time,
                'temperatura_sensor': round(temp_variation, 2),
                'humedad_ambiente': round(hum_variation, 2),
                'nivel_bateria': round(new_battery, 2),
                'tiempo_desde_calibracion': round(new_calibration_time, 2)
            }

    def predict_gas_future(self, hours_ahead, temperatura=None, humedad=None, tiempo_calibracion=None,
                           nivel_bateria=None):
        """
        Predice niveles futuros de gas para las próximas horas especificadas,
        basándose en parámetros iniciales proporcionados.

        Args:
            hours_ahead (int): Número de horas para predecir
            temperatura (float, optional): Temperatura inicial del sensor
            humedad (float, optional): Humedad ambiente inicial
            tiempo_calibracion (float, optional): Tiempo desde la última calibración (horas)
            nivel_bateria (float, optional): Nivel de batería inicial (%)

        Returns:
            dict: Predicciones horarias y metadatos
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado o cargado correctamente.")

        # Usar parámetros proporcionados o valores del último registro
        if all(param is not None for param in [temperatura, humedad, tiempo_calibracion, nivel_bateria]):
            # Usar valores proporcionados
            self.last_known_values = {
                'datetime': datetime.now(),
                'temperatura_sensor': temperatura,
                'humedad_ambiente': humedad,
                'tiempo_desde_calibracion': tiempo_calibracion,
                'nivel_bateria': nivel_bateria
            }
        else:
            # Usar valores del último registro
            self.update_last_known_values()

        predictions = []
        current_datetime = pd.to_datetime(self.last_known_values['datetime'])
        base_values = self.last_known_values.copy()

        # Factor de variación más realista
        variation_factor = random.uniform(0.95, 1.05)  # ±5% de variación

        # Calcular tendencia base del gas basada en el histórico
        if self.df is not None and len(self.df) > 24:
            historical_trend = self.df['nivel_gas_metano'].diff().mean()
        else:
            historical_trend = 0

        for hour in range(hours_ahead):
            future_datetime = current_datetime + timedelta(hours=hour + 1)

            # Variaciones más realistas basadas en el modelo
            hour_of_day = future_datetime.hour
            is_night = 0 <= hour_of_day <= 6  # Factor nocturno

            # Temperatura con ciclo día/noche
            temp_base = base_values['temperatura_sensor']
            temp_cycle = math.sin(hour_of_day * math.pi / 12) * 2
            temp_noise = random.uniform(-0.5, 0.5)
            new_temp = max(0, min(49, temp_base + temp_cycle + temp_noise))

            # Humedad inversa a temperatura
            humidity_base = base_values['humedad_ambiente']
            humidity_cycle = -math.sin(hour_of_day * math.pi / 12) * 1.5
            humidity_noise = random.uniform(-1, 1)
            new_humidity = max(61, min(95, humidity_base + humidity_cycle + humidity_noise))

            # Batería con degradación realista
            battery_drain = hour * random.uniform(0.1, 0.2)
            new_battery = max(10, base_values['nivel_bateria'] - battery_drain)

            # Tiempo de calibración
            new_calibration_time = min(719, base_values['tiempo_desde_calibracion'] + hour)

            prediction_data = {
                'temperatura_sensor': round(new_temp, 2),
                'humedad_ambiente': round(new_humidity, 2),
                'nivel_bateria': round(new_battery, 2),
                'tiempo_desde_calibracion': round(new_calibration_time, 2)
            }

            X_pred = pd.DataFrame([prediction_data], columns=self.X.columns)
            X_pred_scaled = self.scaler.transform(X_pred)

            # Predicción base del modelo
            base_prediction = float(self.model.predict(X_pred_scaled)[0])

            # Ajustes basados en factores temporales y tendencias
            time_factor = 1 - (0.05 if is_night else 0)  # Ligera reducción nocturna
            trend_adjustment = historical_trend * hour  # Tendencia histórica

            # Predicción final con límites
            gas_level = (base_prediction * variation_factor * time_factor + trend_adjustment)
            gas_level_conversion = self.ppm_to_percentage(gas_level)
            gas_level = max(0.01, min(3.73, gas_level_conversion))

            # Análisis de confiabilidad de la predicción
            reliability = self._assess_prediction_reliability(
                prediction_data['temperatura_sensor'],
                prediction_data['humedad_ambiente'],
                prediction_data['tiempo_desde_calibracion'],
                prediction_data['nivel_bateria']
            )

            # Calcular degradación de confiabilidad a lo largo del tiempo
            time_degradation = min(100, (hour / hours_ahead) * 25)  # Máx 25% de degradación por tiempo
            reliability_adjusted = max(0, reliability['overall_reliability_percentage'] - time_degradation)

            predictions.append({
                'datetime': future_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'temperatura_sensor': prediction_data['temperatura_sensor'],
                'humedad_ambiente': prediction_data['humedad_ambiente'],
                'nivel_bateria': prediction_data['nivel_bateria'],
                'tiempo_desde_calibracion': prediction_data['tiempo_desde_calibracion'],
                'predicted_gas_level': round(gas_level, 5),
                'reliability': round(reliability_adjusted, 1)
            })

        return {
            'predictions': predictions,
            'metadata': {
                'total_hours': hours_ahead,
                'start_datetime': current_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'end_datetime': (current_datetime + timedelta(hours=hours_ahead)).strftime('%Y-%m-%d %H:%M:%S'),
                'model_confidence': {
                    'r2_score': round(self.model.score(self.X, self.y), 3),
                    'variation_factor': round(variation_factor, 3),
                    'historical_trend': round(historical_trend, 5) if historical_trend != 0 else 0
                },
                'current_values': base_values
            }
        }

    # Método para hacer la conversión de partes por millón a porcentaje
    def ppm_to_percentage(self, ppm_value):
        """
        Convierte valores de partes por millón (ppm) a porcentaje.
        Acepta tanto valores individuales como listas.

        Args:
            ppm_value: Valor en ppm o lista de valores

        Returns:
            float o lista: Valor(es) convertido(s) a porcentaje
        """
        if isinstance(ppm_value, list):
            return [val / 10000.0 for val in ppm_value]
        elif hasattr(ppm_value, 'tolist'):  # Para arrays de numpy
            return (ppm_value / 10000.0).tolist()
        else:
            return ppm_value / 10000.0

    def percentage_to_ppm(self, percentage_value):
        """
        Convierte valores de porcentaje a partes por millón (ppm).
        Acepta tanto valores individuales como listas.

        Args:
            percentage_value: Valor en porcentaje o lista de valores

        Returns:
            float o lista: Valor(es) convertido(s) a ppm
        """
        if isinstance(percentage_value, list):
            return [val * 10000.0 for val in percentage_value]
        elif hasattr(percentage_value, 'tolist'):  # Para arrays de numpy
            return (percentage_value * 10000.0).tolist()
        else:
            return percentage_value * 10000.0

    ## Predicción de errores de calibración
    def predict_calibration_error(self, temperatura, humedad, tiempo_calibracion, nivel_bateria):
        """
        Predice el error de calibración/deriva del sensor basado en condiciones operacionales.

        Args:
            temperatura (float): Temperatura del sensor (°C)
            humedad (float): Humedad ambiente (%)
            tiempo_calibracion (float): Tiempo desde la última calibración (horas)
            nivel_bateria (float): Nivel de batería (%)

        Returns:
            dict: Predicción del error de calibración y análisis
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado o cargado correctamente.")

        # Factores de deriva individual
        deriva_temporal = self._calculate_temporal_drift(tiempo_calibracion)
        deriva_temperatura = self._calculate_temperature_drift(temperatura)
        deriva_humedad = self._calculate_humidity_drift(humedad)
        deriva_bateria = self._calculate_battery_drift(nivel_bateria)

        # Error total de lectura
        error_calibracion = deriva_temporal + deriva_temperatura + deriva_humedad + deriva_bateria

        # Obtener predicción base
        prediction_result = self.predict(temperatura, humedad, tiempo_calibracion, nivel_bateria)
        nivel_esperado = prediction_result['predicted_gas_level']

        # Conversión a porcentaje
        error_porcentaje = (error_calibracion / 100.0) * 100

        # Evaluación de severidad
        severidad = self._assess_error_severity(abs(error_calibracion))

        return {
            'error_calibracion_absoluto': round(error_calibracion, 3),
            'error_calibracion_porcentaje': round(error_porcentaje, 2),
            'nivel_gas_esperado': nivel_esperado,
            'nivel_gas_con_error': round(nivel_esperado + error_porcentaje, 2),
            'factores_contribuyentes': {
                'deriva_temporal': round(deriva_temporal, 3),
                'deriva_temperatura': round(deriva_temperatura, 3),
                'deriva_humedad': round(deriva_humedad, 3),
                'deriva_bateria': round(deriva_bateria, 3)
            },
            'severidad_error': severidad,
            'recomendacion_calibracion': self._get_calibration_recommendation(error_calibracion, tiempo_calibracion),
            'modelo_tipo': 'Regresión Lineal'
        }

    def predict_reading_uncertainty(self, temperatura, humedad, tiempo_calibracion, nivel_bateria):
        """
        Predice la incertidumbre/confiabilidad de la lectura actual del sensor.

        Args:
            temperatura (float): Temperatura del sensor (°C)
            humedad (float): Humedad ambiente (%)
            tiempo_calibracion (float): Tiempo desde la última calibración (horas)
            nivel_bateria (float): Nivel de batería (%)

        Returns:
            dict: Análisis de incertidumbre y confiabilidad
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado o cargado correctamente.")

        # Obtener predicción base
        prediction_result = self.predict(temperatura, humedad, tiempo_calibracion, nivel_bateria)

        # Calcular incertidumbres
        incertidumbre_calibracion = self._calculate_calibration_uncertainty(tiempo_calibracion)
        incertidumbre_ambiental = self._calculate_environmental_uncertainty(temperatura, humedad)
        incertidumbre_hardware = self._calculate_hardware_uncertainty(nivel_bateria)

        # Incertidumbre total
        incertidumbre_total = np.sqrt(
            incertidumbre_calibracion ** 2 +
            incertidumbre_ambiental ** 2 +
            incertidumbre_hardware ** 2
        )

        # Nivel de confianza
        confianza_porcentaje = max(0, 100 - (incertidumbre_total * 10))

        # Rango de lectura
        nivel_predicho = prediction_result['predicted_gas_level']
        rango_inferior = max(0, nivel_predicho - incertidumbre_total)
        rango_superior = min(100, nivel_predicho + incertidumbre_total)

        # Clasificación
        clasificacion_confianza = self._classify_confidence(confianza_porcentaje)

        return {
            'incertidumbre_total': round(incertidumbre_total, 2),
            'confianza_porcentaje': round(confianza_porcentaje, 1),
            'clasificacion_confianza': clasificacion_confianza,
            'rango_lectura': {
                'valor_central': nivel_predicho,
                'limite_inferior': round(rango_inferior, 2),
                'limite_superior': round(rango_superior, 2),
                'amplitud_rango': round(rango_superior - rango_inferior, 2)
            },
            'fuentes_incertidumbre': {
                'calibracion': round(incertidumbre_calibracion, 3),
                'ambiental': round(incertidumbre_ambiental, 3),
                'hardware': round(incertidumbre_hardware, 3)
            },
            'recomendaciones': self._get_uncertainty_recommendations(incertidumbre_total, confianza_porcentaje),
            'modelo_tipo': 'Regresión Lineal'
        }

    # Métodos auxiliares de cálculo
    def _calculate_temporal_drift(self, tiempo_calibracion):
        """Calcula deriva temporal del sensor."""
        dias_desde_calibracion = tiempo_calibracion / 24.0
        deriva_base = 0.1 * dias_desde_calibracion

        if dias_desde_calibracion > 7:
            deriva_acelerada = (dias_desde_calibracion - 7) * 0.05
            deriva_base += deriva_acelerada

        return deriva_base

    def _calculate_temperature_drift(self, temperatura):
        """Calcula deriva por temperatura."""
        temperatura_optima = 22.5
        desviacion_temp = abs(temperatura - temperatura_optima)
        deriva_temp = desviacion_temp * 0.02
        return deriva_temp

    def _calculate_humidity_drift(self, humedad):
        """Calcula deriva por humedad."""
        if 45 <= humedad <= 75:
            deriva_hum = 0.0
        else:
            if humedad < 45:
                deriva_hum = (45 - humedad) * 0.01
            else:
                deriva_hum = (humedad - 75) * 0.015
        return deriva_hum

    def _calculate_battery_drift(self, nivel_bateria):
        """Calcula deriva por nivel de batería."""
        if nivel_bateria >= 50:
            deriva_bat = 0.0
        elif nivel_bateria >= 30:
            deriva_bat = (50 - nivel_bateria) * 0.02
        else:
            deriva_bat = 0.4 + (30 - nivel_bateria) * 0.05
        return deriva_bat

    def _calculate_calibration_uncertainty(self, tiempo_calibracion):
        """Calcula incertidumbre por calibración."""
        days = tiempo_calibracion / 24.0
        uncertainty = 0.5 + (days * 0.1)
        return min(uncertainty, 5.0)

    def _calculate_environmental_uncertainty(self, temperatura, humedad):
        """Calcula incertidumbre ambiental."""
        temp_uncertainty = abs(temperatura - 22.5) * 0.05
        hum_uncertainty = max(0, abs(humedad - 60) - 15) * 0.02
        return temp_uncertainty + hum_uncertainty

    def _calculate_hardware_uncertainty(self, nivel_bateria):
        """Calcula incertidumbre del hardware."""
        if nivel_bateria >= 50:
            return 0.2
        elif nivel_bateria >= 30:
            return 0.5 + (50 - nivel_bateria) * 0.05
        else:
            return 1.5 + (30 - nivel_bateria) * 0.1

    def _assess_error_severity(self, error_abs):
        """Evalúa severidad del error."""
        if error_abs < 0.5:
            return {'nivel': 'Bajo', 'color': 'verde', 'accion': 'Monitoreo rutinario'}
        elif error_abs < 1.0:
            return {'nivel': 'Moderado', 'color': 'amarillo', 'accion': 'Aumentar frecuencia de verificación'}
        elif error_abs < 2.0:
            return {'nivel': 'Alto', 'color': 'naranja', 'accion': 'Programar calibración pronto'}
        else:
            return {'nivel': 'Crítico', 'color': 'rojo', 'accion': 'Calibración inmediata requerida'}

    def _classify_confidence(self, confianza_porcentaje):
        """Clasifica el nivel de confianza."""
        if confianza_porcentaje >= 90:
            return {'nivel': 'Muy Alta', 'descripcion': 'Lectura altamente confiable'}
        elif confianza_porcentaje >= 75:
            return {'nivel': 'Alta', 'descripcion': 'Lectura confiable'}
        elif confianza_porcentaje >= 60:
            return {'nivel': 'Media', 'descripcion': 'Lectura moderadamente confiable'}
        elif confianza_porcentaje >= 40:
            return {'nivel': 'Baja', 'descripcion': 'Lectura poco confiable'}
        else:
            return {'nivel': 'Muy Baja', 'descripcion': 'Lectura no confiable'}

    def _get_calibration_recommendation(self, error_calibracion, tiempo_calibracion):
        """Genera recomendación de calibración."""
        error_abs = abs(error_calibracion)
        days_since_cal = tiempo_calibracion / 24.0

        if error_abs > 2.0:
            return "Calibración inmediata requerida - Error crítico detectado"
        elif error_abs > 1.0 or days_since_cal > 14:
            return "Programar calibración en las próximas 24-48 horas"
        elif error_abs > 0.5 or days_since_cal > 7:
            return "Considerar calibración en la próxima semana"
        else:
            return "Sensor dentro de especificaciones - Continuar monitoreo rutinario"

    def _get_uncertainty_recommendations(self, incertidumbre_total, confianza_porcentaje):
        """Genera recomendaciones basadas en incertidumbre."""
        recommendations = []

        if incertidumbre_total > 3.0:
            recommendations.append("Incertidumbre alta - Verificar condiciones ambientales")

        if confianza_porcentaje < 60:
            recommendations.append("Confianza baja - Tomar múltiples lecturas para confirmar")

        if confianza_porcentaje < 40:
            recommendations.append("Confianza muy baja - Suspender operaciones críticas hasta calibración")

        if not recommendations:
            recommendations.append("Sensor operando dentro de parámetros aceptables")

        return recommendations

    def get_battery_gas_data(self):
        """
        Retorna los datos reales de nivel de batería y nivel de gas metano del dataset.

        Returns:
            dict: Datos de batería y gas con estructura simple
        """
        try:
            if self.df is None:  # Cambiar 'data' por 'df' que es como se llama en tu modelo
                raise ValueError("No hay datos cargados")

            # Filtrar solo las columnas necesarias y eliminar valores NaN
            battery_gas_data = self.df[['nivel_bateria', 'nivel_gas_metano']].dropna()

            if len(battery_gas_data) == 0:
                raise ValueError("No hay datos válidos de batería y gas")

            # Convertir a lista de diccionarios para fácil manejo en frontend
            data_pairs = []
            for index, row in battery_gas_data.iterrows():
                data_pairs.append({
                    'battery_level': float(row['nivel_bateria']),
                    'gas_level': float(row['nivel_gas_metano']),
                    'index': int(index)
                })

            return {
                'data': data_pairs,
                'total_records': len(data_pairs),
                'battery_range': {
                    'min': float(battery_gas_data['nivel_bateria'].min()),
                    'max': float(battery_gas_data['nivel_bateria'].max())
                },
                'gas_range': {
                    'min': float(battery_gas_data['nivel_gas_metano'].min()),
                    'max': float(battery_gas_data['nivel_gas_metano'].max())
                }
            }

        except Exception as e:
            return {
                'error': str(e),
                'data': [],
                'total_records': 0
            }
