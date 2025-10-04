import pandas as pd
import numpy as np
import random
import math
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from datetime import datetime, timedelta
from prophet import Prophet


class GasLevelProphetModel:
    """
    Modelo para análisis y predicción de niveles de gas metano en sistemas embebidos
    de minas subterráneas usando Prophet, considerando factores operacionales como temperatura,
    humedad, tiempo de calibración y nivel de batería.
    """

    def __init__(self, model_path='./trained_models/prophet_model.pkl'):
        """
        Inicializa el modelo con valores por defecto.
        """
        self.model = None
        self.scaler = None
        self.df = None
        self.X = None
        self.y = None
        self.last_known_values = None
        self.model_path = model_path
        self.training_results = None  # Almacena los resultados del entrenamiento
        self.feature_names = ['temperatura_sensor', 'humedad_ambiente',
                              'tiempo_desde_calibracion', 'nivel_bateria']
        self.correction_history = {}  # Historial de correcciones aplicadas
        self._load_trained_model()

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

        # Actualización de últimos valores conocidos para predicciones
        self.update_last_known_values()

        # Verificación de datos faltantes
        missing_data = self.df.isnull().sum()

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

    def _load_trained_model(self):
        """
        Carga el modelo entrenado y sus resultados si existe.

        Returns:
            bool: True si el modelo se cargó correctamente, False en caso contrario.
        """
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.scaler = saved_data.get('scaler')
                self.training_results = saved_data.get('training_results')
                self.X = saved_data.get('X')
                self.y = saved_data.get('y')
                self.df = saved_data.get('df')
                print("Modelo cargado exitosamente desde", self.model_path)
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
        if self.model is not None:
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
                print("Modelo guardado exitosamente en", self.model_path)
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

    def _calculate_importance_scores(self):
        """
        Calcula la importancia relativa de las variables predictoras.
        Para Prophet, se calculan basándose en el impacto que tienen los regresores.

        Returns:
            dict: Importancia relativa de cada variable
        """
        # Prophet no proporciona directamente feature_importances como GradientBoosting
        # Calculamos con un método alternativo basado en el tamaño de los coeficientes

        # Extraer el nombre de los regresores y sus coeficientes
        importance_scores = {}

        # En Prophet, los regresores existen en self.model.params
        # Tomamos los coeficientes de los regresores
        if hasattr(self.model, 'params') and 'extra_regressors' in self.model.params:
            for regressor_name, regressor_params in self.model.params['extra_regressors'].items():
                if 'beta' in regressor_params and regressor_params['beta'] is not None:
                    importance_scores[regressor_name] = abs(regressor_params['beta'])

        # Normalizar para que sumen 1, similar a GradientBoostingRegressor
        if importance_scores:
            total = sum(importance_scores.values())
            if total > 0:
                importance_scores = {k: v / total for k, v in importance_scores.items()}

        # Si no hay datos en el modelo, proporcionamos estimaciones basadas en correlaciones
        if not importance_scores and self.df is not None:
            cors = {}
            for feature in self.feature_names:
                cors[feature] = abs(self.df[[feature, 'nivel_gas_metano']].corr().iloc[0, 1])

            total = sum(cors.values())
            if total > 0:
                importance_scores = {k: v / total for k, v in cors.items()}
            else:
                # Si no hay correlaciones, distribuimos equitativamente
                importance_scores = {feature: 1.0 / len(self.feature_names) for feature in self.feature_names}

        return importance_scores

    def _calculate_training_results(self):
        """
        Calcula las métricas de rendimiento del modelo entrenado.
        """
        if self.model is None:
            raise ValueError("El modelo no está entrenado.")

        # Preparar datos para evaluación
        train_data = pd.DataFrame({
            'ds': self.df['datetime'],
            'y': self.y
        })

        # Añadir regresores
        for feature in self.feature_names:
            train_data[feature] = self.df[feature]

        # Dividir datos para evaluación
        train_indices, test_indices = train_test_split(
            range(len(train_data)), test_size=0.2, random_state=42
        )

        train_df = train_data.iloc[train_indices]
        test_df = train_data.iloc[test_indices]

        # Predecir con el modelo en datos de prueba
        test_pred = self.model.predict(test_df)

        # Calcular métricas
        y_test = test_df['y'].values
        y_pred = test_pred['yhat'].values

        # Conversion de las metricas a porcentaje
        metric_y_test = self.ppm_to_percentage(y_test)
        metric_y_pred = self.ppm_to_percentage(y_pred)

        # Calcular importancia de características
        feature_importance = self._calculate_importance_scores()

        # Convertir a lista manteniendo el orden de self.feature_names
        importances = [feature_importance.get(feature, 0.0) for feature in self.feature_names]

        # Residuales
        residuals_values = (y_test - y_pred).tolist()
        residuals_predictions = y_pred.tolist()

        self.training_results = {
            'metrics': {
                'mse': mean_squared_error(metric_y_test, metric_y_pred),
                'rmse': np.sqrt(mean_squared_error(metric_y_test, metric_y_pred)),
                'mae': mean_absolute_error(metric_y_test, metric_y_pred),
                'r2': r2_score(metric_y_test, metric_y_pred)
            },
            'prediction_data': {
                'real_values': self.ppm_to_percentage(y_test.tolist()),
                'predicted_values': self.ppm_to_percentage(y_pred.tolist())
                # 'real_values': y_test.tolist(),
                # 'predicted_values': y_pred.tolist()
            },
            'feature_importance': {
                'variables': self.feature_names,
                'importances': self.ppm_to_percentage(importances)
            },
            'residuals': {
                'values': self.ppm_to_percentage(residuals_values),
                'predictions': self.ppm_to_percentage(residuals_predictions)
            },
            'prophet_components': {
                'trend': [float(x) for x in test_pred['trend'].values],
                'seasonality': [float(x) for x in (test_pred['yhat'] - test_pred['trend']).values[:10]]
                # Muestra reducida
            }
        }

    def train_model(self, force_retrain=False):
        """
        Entrena el modelo Prophet para predecir niveles de gas, aplicando
        preprocesamiento de datos para manejar outliers y valores faltantes.

        Args:
            force_retrain (bool): Si es True, fuerza el reentrenamiento incluso si ya existe

        Returns:
            dict: Métricas de evaluación y datos de predicción
        """
        # Valida si ya existe un modelo entrenado
        if self.model is not None and not force_retrain:
            return self.get_training_results()

        if self.df is None:
            return {"error": "No data loaded. Call load_data() first."}

        # Procesar columnas principales para análisis y preprocesamiento
        columns_to_process = [
            'temperatura_sensor',
            'humedad_ambiente',
            'nivel_gas_metano',
            'nivel_bateria'
        ]

        # Detectar outliers
        outliers_info = self.detect_outliers(columns_to_process)

        # Imputar valores faltantes
        # imputations_info = self.impute_missing_values(
        #     columns=columns_to_process,
        #     method='median'  # La mediana suele ser robusta para datos asimétricos
        # )

        # Corregir datos atípicos
        # corrections_info = self.correct_outliers(
        #     columns=columns_to_process,
        #     threshold=3,
        #     method='median'  # La mediana es generalmente más robusta para outliers
        # )

        # Definir variables predictoras (X) y variable objetivo (y)
        self.X = self.df[self.feature_names]
        self.y = self.df['nivel_gas_metano']

        # Normalizar variables predictoras (opcional para Prophet pero útil para análisis)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        # Preparar datos para Prophet
        train_data = pd.DataFrame({
            'ds': self.df['datetime'],
            'y': self.y
        })

        # Añadir variables predictoras como regresores
        for feature in self.feature_names:
            train_data[feature] = self.df[feature]

        # Inicializar y entrenar modelo Prophet
        self.model = Prophet(
            yearly_seasonality=True,  # Detectar patrones anuales
            weekly_seasonality=True,  # Detectar patrones semanales
            daily_seasonality=True,  # Detectar patrones diarios
            changepoint_prior_scale=0.05,  # Flexibilidad para cambios de tendencia
            seasonality_prior_scale=10,  # Flexibilidad para efectos estacionales
            seasonality_mode='multiplicative',  # Estacionalidad que varía con el nivel
            interval_width=0.95  # Intervalo de confianza del 95%
        )

        # Añadir regresores
        for feature in self.feature_names:
            self.model.add_regressor(feature)

        # Entrenar modelo
        self.model.fit(train_data)

        # Calcular y guardar resultados del entrenamiento
        self._calculate_training_results()

        # Guardar el modelo entrenado
        self._save_trained_model()

        # Añadir información de preproces
        self.training_results['preprocessing'] = {
            'outliers': outliers_info,
            'imputations': imputations_info,
            'corrections': corrections_info
        }

        # Añadir componentes de Prophet a los resultados
        forecast = self.model.predict(train_data)
        self.training_results['prophet_components'] = {
            'trend_description': 'Tendencia general de los niveles de gas a lo largo del tiempo',
            'yearly_seasonality': self.model.seasonalities.get('yearly', {}).get('condition', False),
            'weekly_seasonality': self.model.seasonalities.get('weekly', {}).get('condition', False),
            'daily_seasonality': self.model.seasonalities.get('daily', {}).get('condition', False),
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
        current_date = datetime.now()
        future = pd.DataFrame({
            'ds': [current_date],
            'temperatura_sensor': [temperatura],
            'humedad_ambiente': [humedad],
            'tiempo_desde_calibracion': [tiempo_calibracion],
            'nivel_bateria': [nivel_bateria]
        })

        # Realizar predicción con Prophet
        forecast = self.model.predict(future)
        prediction = float(forecast.iloc[0]['yhat'])

        # Obtener intervalos de confianza para el análisis de confiabilidad
        lower_bound = float(forecast.iloc[0]['yhat_lower'])
        upper_bound = float(forecast.iloc[0]['yhat_upper'])

        # Calcular incertidumbre como porcentaje del valor predicho
        uncertainty_range = upper_bound - lower_bound
        uncertainty_percent = (uncertainty_range / prediction) * 100 if prediction != 0 else 0

        # Añadir análisis de confiabilidad
        reliability_analysis = self._assess_prediction_reliability(
            temperatura, humedad, tiempo_calibracion, nivel_bateria,
            uncertainty_percent=uncertainty_percent
        )

        return {
            'predicted_gas_level': self.ppm_to_percentage(prediction),
            'reliability_analysis': reliability_analysis,
            'confidence_interval': {
                'lower': self.ppm_to_percentage(lower_bound),
                'upper': self.ppm_to_percentage(upper_bound)
            }
        }

    def _assess_prediction_reliability(self, temperatura, humedad, tiempo_calibracion, nivel_bateria,
                                       uncertainty_percent=None):
        """
        Evalúa la confiabilidad de una predicción basada en los valores de entrada.

        Args:
            temperatura (float): Temperatura del sensor
            humedad (float): Humedad ambiente
            tiempo_calibracion (float): Tiempo desde la última calibración (horas)
            nivel_bateria (float): Nivel de batería (%)
            uncertainty_percent (float, optional): Porcentaje de incertidumbre de Prophet

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

        # Factor adicional específico de Prophet - incertidumbre del modelo
        if uncertainty_percent is not None:
            if uncertainty_percent > 30:
                reliability_factors['uncertainty'] = {
                    'factor': 'Alta incertidumbre en la predicción del modelo',
                    'impact': 'alto',
                    'reduction': 25
                }
                overall_reliability -= 25
            elif uncertainty_percent > 15:
                reliability_factors['uncertainty'] = {
                    'factor': 'Incertidumbre moderada en la predicción del modelo',
                    'impact': 'medio',
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
        Esta función simula cambios realistas en los valores del sistema embebido
        a lo largo del tiempo para proporcionar datos iniciales para predicciones.
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
        basándose en parámetros iniciales proporcionados y utilizando el modelo Prophet.

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
            raise ValueError("Modelo no entrenado. Llame a train_model() primero.")

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

        # Crear dataframe para proyecciones futuras
        future_dates = pd.date_range(start=current_datetime, periods=hours_ahead + 1, freq='H')[1:]
        future_df = pd.DataFrame({'ds': future_dates})

        # Proyectar valores de los factores operacionales a futuro
        for hour, date in enumerate(future_dates):
            hour_of_day = date.hour
            is_night = 0 <= hour_of_day <= 6

            # Cálculos incrementales para cada factor
            # Temperatura con ciclo día/noche
            if hour == 0:
                base_temp = self.last_known_values['temperatura_sensor']
            else:
                base_temp = float(future_df.loc[hour - 1, 'temperatura_sensor'])

            temp_cycle = math.sin(hour_of_day * math.pi / 12) * 1.5
            temp_noise = random.uniform(-0.3, 0.3)
            new_temp = max(0, min(49, base_temp + temp_cycle + temp_noise))
            future_df.loc[hour, 'temperatura_sensor'] = round(new_temp, 2)

            # Humedad con relación inversa a temperatura
            if hour == 0:
                base_humidity = self.last_known_values['humedad_ambiente']
            else:
                base_humidity = float(future_df.loc[hour - 1, 'humedad_ambiente'])

            humidity_cycle = -math.sin(hour_of_day * math.pi / 12) * 1.2
            humidity_noise = random.uniform(-0.8, 0.8)
            new_humidity = max(61, min(95, base_humidity + humidity_cycle + humidity_noise))
            future_df.loc[hour, 'humedad_ambiente'] = round(new_humidity, 2)

            # Tiempo desde calibración
            if hour == 0:
                base_calibration = self.last_known_values['tiempo_desde_calibracion']
            else:
                base_calibration = float(future_df.loc[hour - 1, 'tiempo_desde_calibracion'])

            new_calibration_time = min(719, base_calibration + 1)  # Incremento de 1 hora
            future_df.loc[hour, 'tiempo_desde_calibracion'] = round(new_calibration_time, 2)

            # Degradación de batería
            if hour == 0:
                base_battery = self.last_known_values['nivel_bateria']
            else:
                base_battery = float(future_df.loc[hour - 1, 'nivel_bateria'])

            battery_drain = random.uniform(0.08, 0.15)
            new_battery = max(10, base_battery - battery_drain)
            future_df.loc[hour, 'nivel_bateria'] = round(new_battery, 2)

        # Hacer predicción con Prophet
        forecast = self.model.predict(future_df)

        # Extraer resultados y calcular confiabilidad para cada hora
        for hour in range(hours_ahead):
            row = forecast.iloc[hour]
            date = future_dates[hour]

            # Convertir predicción a porcentaje y aplicar límites
            gas_level = float(row['yhat'])
            gas_level_percentage = self.ppm_to_percentage(gas_level)
            gas_level_percentage = max(0.01, min(1.5, gas_level_percentage))

            # Calcular incertidumbre para confiabilidad
            uncertainty_percent = (row['yhat_upper'] - row['yhat_lower']) / row['yhat'] * 100 if row['yhat'] != 0 else 0

            # Factores operacionales para esta hora
            temp = future_df.loc[hour, 'temperatura_sensor']
            humidity = future_df.loc[hour, 'humedad_ambiente']
            calibration_time = future_df.loc[hour, 'tiempo_desde_calibracion']
            battery = future_df.loc[hour, 'nivel_bateria']

            # Calcular confiabilidad de la predicción
            reliability_base = self._assess_prediction_reliability(
                temp, humidity, calibration_time, battery,
                uncertainty_percent=uncertainty_percent
            )

            # Degradación de confiabilidad por tiempo
            time_degradation = min(100, (hour / hours_ahead) * 15)  # Max 15% degradación por tiempo
            reliability_adjusted = max(0, reliability_base['overall_reliability_percentage'] - time_degradation)

            predictions.append({
                'datetime': date.strftime('%Y-%m-%d %H:%M:%S'),
                'temperatura_sensor': float(temp),
                'humedad_ambiente': float(humidity),
                'nivel_bateria': float(battery),
                'tiempo_desde_calibracion': float(calibration_time),
                'predicted_gas_level': round(gas_level_percentage, 5),
                'reliability': round(reliability_adjusted, 1),
                'lower_bound': round(self.ppm_to_percentage(float(row['yhat_lower'])), 3),
                'upper_bound': round(self.ppm_to_percentage(float(row['yhat_upper'])), 3)
            })

        # Información del modelo para metadatos
        model_info = {
            'seasonalities': list(self.model.seasonalities.keys()),
            'n_changepoints': len(self.model.changepoints) if hasattr(self.model, 'changepoints') else 0,
            'regressor_coefficients': {
                feat: self.model.params['extra_regressors'][feat]['beta']
                for feat in self.feature_names
                if feat in self.model.params.get('extra_regressors', {})
            }
        }

        return {
            'predictions': predictions,
            'metadata': {
                'total_hours': hours_ahead,
                'start_datetime': current_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'end_datetime': (current_datetime + timedelta(hours=hours_ahead)).strftime('%Y-%m-%d %H:%M:%S'),
                'model_confidence': {
                    'prophet_info': model_info,
                    'prophet_description': 'Prophet captura tendencia, estacionalidad y efectos de regresores externos'
                },
                'current_values': self.last_known_values
            }
        }

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