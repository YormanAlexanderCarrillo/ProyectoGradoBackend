import pandas as pd
import numpy as np
import random
import math
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from datetime import datetime, timedelta



class GasLevelGradientBoostingModel:
    def __init__(self, model_path='./trained_models/gradient_boosting_model.pkl'):
        self.model = None
        self.scaler = None
        self.df = None
        self.X = None
        self.y = None
        self.last_known_values = None
        self.model_path = model_path
        self.training_results = None  # Almacena los resultados del entrenamiento
        self._load_trained_model()

    def load_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['datetime'] = pd.to_datetime(self.df['fecha'] + ' ' + self.df['hora'])
        self.df['dias_desde_calibracion'] = self.df['tiempo_desde_calibracion'] / 24
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

    def _load_trained_model(self):
        """Intenta cargar un modelo previamente entrenado y sus resultados si existen."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.training_results = saved_data.get('training_results')
                print("Modelo cargado exitosamente desde", self.model_path)
                return True
            except Exception as e:
                print(f"Error al cargar el modelo: {str(e)}")
                self.model = None
                self.scaler = None
                self.training_results = None
        return False

    def _save_trained_model(self):
        """Guarda el modelo entrenado, el scaler y los resultados del entrenamiento."""
        if self.model is not None and self.scaler is not None:
            try:
                with open(self.model_path, 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'scaler': self.scaler,
                        'training_results': self.training_results
                    }, f)
                print("Modelo guardado exitosamente en", self.model_path)
                return True
            except Exception as e:
                print(f"Error al guardar el modelo: {str(e)}")
        return False

    def get_training_results(self):
        """Retorna los resultados del entrenamiento sin reentrenar."""

        print("usted si debe venir aca")

        if self.training_results is None:
            if self.model is None:
                raise ValueError("El modelo no está entrenado. Ejecute train_model() primero.")
            # Si no hay resultados guardados pero el modelo existe, los recalculamos una vez
            self._calculate_training_results()
        return self.training_results

    def _calculate_training_results(self):

        print("mas o menos usted deberia estar aca")

        """Calcula las métricas y resultados del modelo sin reentrenar."""
        if self.model is None or self.scaler is None:
            raise ValueError("El modelo no está entrenado.")

        X_scaled = self.scaler.transform(self.X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42
        )

        y_pred = self.model.predict(X_test)

        self.training_results = {
            'metrics': {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            },
            'prediction_data': {
                'real_values': y_test.tolist(),
                'predicted_values': y_pred.tolist()
            },
            'feature_importance': {
                'variables': self.X.columns.tolist(),
                'importances': self.model.feature_importances_.tolist()
            },
            'residuals': {
                'values': (y_test - y_pred).tolist(),
                'predictions': y_pred.tolist()
            }
        }

    def train_model(self, force_retrain=False):
        """
        Entrena el modelo solo si es necesario y guarda los resultados.
        """

        print("Usted que hace aqui")

        if self.model is not None and not force_retrain:
            return self.get_training_results()

        if self.X is None or self.y is None:
            if self.df is None:
                raise ValueError("No hay datos cargados. Llame a load_data() primero.")

            self.X = self.df[['temperatura_sensor', 'humedad_ambiente',
                              'tiempo_desde_calibracion', 'nivel_bateria']]
            self.y = self.df['nivel_gas_metano']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42
        )

        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        # Calcular y guardar los resultados del entrenamiento
        self._calculate_training_results()

        # Guardar todo
        self._save_trained_model()

        return self.training_results

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

    def update_last_known_values(self):
        if self.df is not None and not self.df.empty:
            latest_row = self.df.iloc[-1]
            current_time = datetime.now()

            temp_current = latest_row['temperatura_sensor']
            temp_variation = max(0, min(49, temp_current + random.uniform(-2, 2)))

            hum_current = latest_row['humedad_ambiente']
            hum_variation = max(61, min(95, hum_current + random.uniform(-3, 3)))

            battery_drain = random.uniform(0.1, 0.3)
            new_battery = max(90, min(100, latest_row['nivel_bateria'] - battery_drain))

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

        # Factor de variación con mayor peso en la tendencia del modelo
        variation_factor = random.uniform(0.97, 1.03)  # ±3% de variación (más conservador)

        # Calcular tendencia histórica con mayor peso en datos recientes
        if self.df is not None and len(self.df) > 24:
            recent_data = self.df.tail(24)  # Últimas 24 horas
            historical_trend = recent_data['nivel_gas_metano'].diff().mean()
        else:
            historical_trend = 0

        for hour in range(hours_ahead):
            future_datetime = current_datetime + timedelta(hours=hour + 1)
            hour_of_day = future_datetime.hour
            is_night = 0 <= hour_of_day <= 6

            # Temperatura con ciclo día/noche más suave
            temp_base = base_values['temperatura_sensor']
            temp_cycle = math.sin(hour_of_day * math.pi / 12) * 1.5
            temp_noise = random.uniform(-0.3, 0.3)
            new_temp = max(0, min(49, temp_base + temp_cycle + temp_noise))

            # Humedad con relación inversa a temperatura
            humidity_base = base_values['humedad_ambiente']
            humidity_cycle = -math.sin(hour_of_day * math.pi / 12) * 1.2
            humidity_noise = random.uniform(-0.8, 0.8)
            new_humidity = max(61, min(95, humidity_base + humidity_cycle + humidity_noise))

            # Degradación de batería más realista
            battery_drain = hour * random.uniform(0.08, 0.15)
            new_battery = max(10, base_values['nivel_bateria'] - battery_drain)

            new_calibration_time = min(719, base_values['tiempo_desde_calibracion'] + hour)

            prediction_data = {
                'temperatura_sensor': round(new_temp, 2),
                'humedad_ambiente': round(new_humidity, 2),
                'nivel_bateria': round(new_battery, 2),
                'tiempo_desde_calibracion': round(new_calibration_time, 2)
            }

            X_pred = pd.DataFrame([prediction_data], columns=self.X.columns)
            X_pred_scaled = self.scaler.transform(X_pred)

            # Predicción base usando Gradient Boosting
            base_prediction = float(self.model.predict(X_pred_scaled)[0])

            # Ajustes más suaves basados en factores temporales
            time_factor = 1 - (0.03 if is_night else 0)  # Reducción nocturna más sutil
            trend_adjustment = historical_trend * hour * 0.8  # Menor peso de la tendencia histórica

            # Predicción final con límites ajustados
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
                    'feature_importances': dict(zip(self.X.columns, self.model.feature_importances_)),
                    'variation_factor': round(variation_factor, 3),
                    'historical_trend': round(historical_trend, 5) if historical_trend != 0 else 0
                },
                'current_values': base_values
            }
        }