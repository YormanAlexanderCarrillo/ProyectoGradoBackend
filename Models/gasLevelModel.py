import pandas as pd
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from datetime import datetime, timedelta

##para cargar y manipular los datos se usan las librerias (pandas y numpy)
## Dividir los datos se usan las librerias ( train_test_split)
# aplicar el modelo de regresión lineal (LinearRegression)
# normalizar los datos (StandardScaler)
# calcular métricas de evaluación (mean_squared_error, r2_score, mean_absolute_error)
# detectar valores atípicos (stats de scipy).

## modelo de regresion lineal

class GasLevelModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.df = None
        self.X = None
        self.y = None
        self.last_known_values = None

    def load_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['datetime'] = pd.to_datetime(self.df['fecha'] + ' ' + self.df['hora'])
        self.df['dias_desde_calibracion'] = self.df['tiempo_desde_calibracion'] / 24
        # el metodo get_basic_stats() retorna las estadísticas descriptivas del conjunto de datos.
        self.update_last_known_values()
        return self.get_basic_stats()

    def get_basic_stats(self):
        return {
            "descriptive_stats": self.df.describe().replace({np.nan: None}).to_dict(),
            'data_shape': self.df.shape
        }

    def detect_outliers(self, columns, threshold=3):
        outliers = {}
        for column in columns:
            z_scores = np.abs(stats.zscore(self.df[column]))
            outlier_indices = np.where(z_scores > threshold)[0]
            outliers[column] = {
                'count': len(outlier_indices),
                'values': self.df.loc[outlier_indices, column].tolist(),
                'indices': outlier_indices.tolist()
            }
        return {
            "outliers": outliers
        }

    def analyze_temporal_degradation(self):
        error_by_day = self.df.groupby('dias_desde_calibracion').agg({
            'nivel_gas_metano': ['mean', 'std']
        }).reset_index()

        return {
            'days': error_by_day['dias_desde_calibracion'].tolist(),
            'std_dev': error_by_day['nivel_gas_metano']['std'].replace({np.nan: None}).tolist(),
            'mean': error_by_day['nivel_gas_metano']['mean'].tolist()
        }

    def get_correlations(self):
        columns = ['temperatura_sensor', 'humedad_ambiente',
                   'tiempo_desde_calibracion', 'nivel_gas_metano', 'nivel_bateria']
        return {
            "correlations": self.df[columns].corr().to_dict()
        }


    def train_model(self):
        self.X = self.df[['temperatura_sensor', 'humedad_ambiente',
                          'tiempo_desde_calibracion', 'nivel_bateria']]
        self.y = self.df['nivel_gas_metano']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42
        )

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        # Datos para gráfico de predicciones vs reales
        prediction_data = {
            'real_values': y_test.tolist(),
            'predicted_values': y_pred.tolist()
        }

        # Importancia de variables
        feature_importance = {
            'variables': self.X.columns.tolist(),
            'coefficients': self.model.coef_.tolist()
        }

        # Residuos
        residuals = {
            'values': (y_test - y_pred).tolist(),
            'predictions': y_pred.tolist()
        }

        return {
            'metrics': metrics,
            'prediction_data': prediction_data,
            'feature_importance': feature_importance,
            'residuals': residuals
        }

    def predict(self, temperatura, humedad, tiempo_calibracion, nivel_bateria):
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llame a train_model() primero.")

        nuevos_datos = pd.DataFrame([[temperatura, humedad, tiempo_calibracion, nivel_bateria]],
                                    columns=self.X.columns)
        nuevos_datos_scaled = pd.DataFrame(
            self.scaler.transform(nuevos_datos),
            columns=self.X.columns
        )
        return float(self.model.predict(nuevos_datos_scaled)[0])

    #time series temporal

    def update_last_known_values(self):
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
            new_battery = max(90, min(100, latest_row['nivel_bateria'] - battery_drain))

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

    def predict_gas_future(self, hours_ahead):
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llame a train_model() primero.")

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
            gas_level = max(0.01, min(3.73, gas_level))

            predictions.append({
                'datetime': future_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'temperatura_sensor': prediction_data['temperatura_sensor'],
                'predicted_gas_level': round(gas_level, 3),
                'nivel_bateria': prediction_data['nivel_bateria'],
                'tiempo_desde_calibracion': prediction_data['tiempo_desde_calibracion'],
                'humedad_ambiente': prediction_data['humedad_ambiente']
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
