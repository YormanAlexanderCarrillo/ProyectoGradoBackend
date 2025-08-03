import numpy as np
import pandas as pd
import pickle
import os
from flask import jsonify
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import time
import warnings



class ComparativeModels:
    def __init__(self):
        """
        Inicializa el controlador cargando los modelos pre-entrenados.
        """
        self.models = {}
        self.data_path = './data/complete_data_normal.csv'
        self.models_path = './trained_models'

        # Mapeo de nombres de modelos a archivos
        self.model_files = {
            'regresion_lineal': 'linear_regression_model.pkl',
            'random_forest': 'random_forest_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl',
            'prophet': 'prophet_model.pkl'
        }

        # Cargar datos una sola vez
        self.X, self.y = self._load_data()

        # Cargar modelos pre-entrenados
        self._load_pretrained_models()

    def _load_data(self):
        """
        Carga los datos desde el archivo CSV.

        Returns:
            tuple: (X, y) características y variable objetivo
        """
        try:
            df = pd.read_csv(self.data_path)

            # Procesar fechas y tiempos si es necesario
            if 'fecha' in df.columns and 'hora' in df.columns:
                df['datetime'] = pd.to_datetime(df['fecha'] + ' ' + df['hora'])

            # Definir características
            feature_names = ['temperatura_sensor', 'humedad_ambiente',
                             'tiempo_desde_calibracion', 'nivel_bateria']

            X = df[feature_names]
            y = df['nivel_gas_metano']

            print(f"Datos cargados exitosamente: {len(df)} registros")
            return X, y

        except Exception as e:
            print(f"Error cargando datos: {str(e)}")
            return None, None

    def _load_pretrained_models(self):
        """
        Carga los modelos pre-entrenados desde archivos .pkl
        """
        for model_name, filename in self.model_files.items():
            file_path = os.path.join(self.models_path, filename)

            try:
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        model_data = pickle.load(f)

                    self.models[model_name] = model_data
                    print(f"Modelo {model_name} cargado exitosamente desde {filename}")
                else:
                    print(f"Archivo no encontrado: {file_path}")

            except Exception as e:
                print(f"Error cargando modelo {model_name}: {str(e)}")

    def _predict_with_model(self, model_name, model_data, X_test):
        """
        Realiza predicciones con un modelo específico.

        Args:
            model_name (str): Nombre del modelo
            model_data: Datos del modelo cargado
            X_test (pd.DataFrame): Datos de prueba

        Returns:
            np.array: Predicciones
        """
        try:
            if model_name == 'regresion_lineal':
                # Para regresión lineal
                model = model_data['model']
                scaler = model_data.get('scaler')

                if scaler:
                    X_scaled = scaler.transform(X_test)
                    predictions = model.predict(X_scaled)
                else:
                    predictions = model.predict(X_test)

                # Convertir de ppm a porcentaje si es necesario
                predictions = predictions / 10000.0

            elif model_name == 'random_forest':
                # Para Random Forest
                model = model_data['model']
                scaler = model_data.get('scaler')

                if scaler:
                    X_scaled = scaler.transform(X_test)
                    predictions = model.predict(X_scaled)
                else:
                    predictions = model.predict(X_test)

                predictions = predictions / 10000.0

            elif model_name == 'gradient_boosting':
                # Para Gradient Boosting
                model = model_data['model']
                scaler = model_data.get('scaler')

                if scaler:
                    X_scaled = scaler.transform(X_test)
                    predictions = model.predict(X_scaled)
                else:
                    predictions = model.predict(X_test)

                predictions = predictions / 10000.0


            elif model_name == 'prophet':

                # OPTIMIZADO: Predicción en batch para Prophet

                model = model_data['model']

                # Crear DataFrame completo de una vez con todas las filas

                future_data = pd.DataFrame({

                    'ds': pd.Timestamp.now(),  # Mismo timestamp para todos

                    'temperatura_sensor': X_test['temperatura_sensor'].values,

                    'humedad_ambiente': X_test['humedad_ambiente'].values,

                    'tiempo_desde_calibracion': X_test['tiempo_desde_calibracion'].values,

                    'nivel_bateria': X_test['nivel_bateria'].values

                })

                try:

                    # UNA sola llamada a predict con todo el DataFrame

                    forecast = model.predict(future_data)

                    predictions = forecast['yhat'].values / 10000.0

                except:

                    predictions = np.full(len(X_test), 0.5)

                predictions = np.array(predictions)

            return predictions

        except Exception as e:
            print(f"Error en predicción con {model_name}: {str(e)}")
            # Retornar predicciones por defecto en caso de error
            return np.full(len(X_test), 0.5)

    def detect_outliers_iqr(self, data, column):
        """
        Detecta valores atípicos utilizando el método del rango intercuartílico (IQR).
        """
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)].index.tolist()
        return outliers

    def evaluate_model_robustness(self, model_name, model_data, X_test, y_test):
        """
        Evalúa la robustez del modelo ante datos anómalos.
        """
        try:
            # Predicciones con datos originales
            y_pred_original = self._predict_with_model(model_name, model_data, X_test)
            mse_original = mean_squared_error(y_test, y_pred_original)

            # Detectar outliers en las características
            combined_data = X_test.copy()
            combined_data['nivel_gas_metano'] = y_test

            outlier_indices = set()
            for column in X_test.columns:
                column_outliers = self.detect_outliers_iqr(combined_data, column)
                outlier_indices.update(column_outliers)

            # Evaluar con datos sin outliers si hay suficientes datos
            if len(outlier_indices) > 0 and len(X_test) - len(outlier_indices) > 10:
                clean_indices = [i for i in X_test.index if i not in outlier_indices]
                X_test_clean = X_test.loc[clean_indices]
                y_test_clean = y_test.loc[clean_indices]

                y_pred_clean = self._predict_with_model(model_name, model_data, X_test_clean)
                mse_clean = mean_squared_error(y_test_clean, y_pred_clean)

                # Calcular score de robustez (0-1, donde 1 es más robusto)
                robustness_score = 1 - min(abs(mse_original - mse_clean) / max(mse_original, 0.001), 1)
            else:
                robustness_score = 1.0  # Sin outliers suficientes para evaluar

            return {
                'robustness_score': round(robustness_score, 4),
                'outliers_detected': len(outlier_indices),
                'outliers_percentage': round((len(outlier_indices) / len(X_test)) * 100, 2),
                'mse_with_outliers': round(mse_original, 6),
                'evaluation_successful': True
            }

        except Exception as e:
            return {
                'robustness_score': 0.0,
                'outliers_detected': 0,
                'outliers_percentage': 0.0,
                'mse_with_outliers': 0.0,
                'evaluation_successful': False,
                'error': str(e)
            }

    def compare_all_models(self):
        """
        Compara todos los modelos utilizando las métricas definidas.
        """
        try:
            if self.X is None or self.y is None:
                return {
                    'success': False,
                    'error': 'No hay datos cargados'
                }

            if not self.models:
                return {
                    'success': False,
                    'error': 'No hay modelos cargados'
                }

            # División de datos para pruebas (80% entrenamiento, 20% prueba)
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            # Convertir y_test a porcentaje para comparación consistente
            y_test_percentage = y_test / 10000.0

            comparison_results = {
                'models_performance': {},
                'summary_metrics': {},
                'chart_data': {
                    'mse_comparison': [],
                    'mae_comparison': [],
                    'r2_comparison': [],
                    'robustness_comparison': [],
                    'inference_time_comparison': []
                },
                'test_data_info': {
                    'total_samples': len(self.X),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'features': list(self.X.columns)
                }
            }

            for model_name, model_data in self.models.items():
                try:
                    print(f"Evaluando modelo: {model_name}")

                    # Medir tiempo de inferencia
                    start_time = time.time()
                    y_pred = self._predict_with_model(model_name, model_data, X_test)
                    inference_time = time.time() - start_time

                    # Calcular métricas principales
                    mse = mean_squared_error(y_test_percentage, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test_percentage, y_pred)
                    r2 = r2_score(y_test_percentage, y_pred)

                    # Evaluar robustez
                    robustness_metrics = self.evaluate_model_robustness(model_name, model_data, X_test,
                                                                        y_test_percentage)
                    # Calcular métricas adicionales
                    prediction_variance = np.var(y_pred)
                    mean_prediction = np.mean(y_pred)

                    # Almacenar resultados del modelo
                    model_results = {
                        'mse': round(mse, 6),
                        'rmse': round(rmse, 6),
                        'mae': round(mae, 6),
                        'r2': round(r2, 4),
                        'robustness_score': robustness_metrics['robustness_score'],
                        'inference_time_ms': round(inference_time * 1000, 2),
                        'prediction_variance': round(prediction_variance, 6),
                        'mean_prediction': round(mean_prediction, 6),
                        'outliers_detected': robustness_metrics['outliers_detected'],
                        'outliers_percentage': robustness_metrics['outliers_percentage'],
                        'evaluation_details': robustness_metrics
                    }

                    comparison_results['models_performance'][model_name] = model_results

                    # Preparar datos para gráficos
                    model_display_name = model_name.replace('_', ' ').title()

                    comparison_results['chart_data']['mse_comparison'].append({
                        'model': model_display_name,
                        'value': round(mse, 6),
                        'label': f"MSE: {mse:.6f}"
                    })

                    comparison_results['chart_data']['mae_comparison'].append({
                        'model': model_display_name,
                        'value': round(mae, 6),
                        'label': f"MAE: {mae:.6f}"
                    })

                    comparison_results['chart_data']['r2_comparison'].append({
                        'model': model_display_name,
                        'value': round(r2, 4),
                        'label': f"R²: {r2:.4f}"
                    })

                    comparison_results['chart_data']['robustness_comparison'].append({
                        'model': model_display_name,
                        'value': round(robustness_metrics['robustness_score'], 4),
                        'label': f"Robustez: {robustness_metrics['robustness_score']:.4f}"
                    })

                    comparison_results['chart_data']['inference_time_comparison'].append({
                        'model': model_display_name,
                        'value': round(inference_time * 1000, 2),
                        'label': f"Tiempo: {inference_time * 1000:.2f}ms"
                    })

                    print(f"Modelo {model_name} evaluado exitosamente")

                except Exception as e:
                    print(f"Error evaluando modelo {model_name}: {str(e)}")
                    comparison_results['models_performance'][model_name] = {
                        'error': str(e),
                        'evaluation_failed': True
                    }

            # Calcular métricas de resumen
            successful_models = {k: v for k, v in comparison_results['models_performance'].items()
                                 if 'error' not in v}

            if successful_models:
                mse_values = [v['mse'] for v in successful_models.values()]
                mae_values = [v['mae'] for v in successful_models.values()]
                r2_values = [v['r2'] for v in successful_models.values()]
                robustness_values = [v['robustness_score'] for v in successful_models.values()]

                comparison_results['summary_metrics'] = {
                    'best_mse_model': min(successful_models.items(), key=lambda x: x[1]['mse'])[0],
                    'best_mae_model': min(successful_models.items(), key=lambda x: x[1]['mae'])[0],
                    'best_r2_model': max(successful_models.items(), key=lambda x: x[1]['r2'])[0],
                    'best_robustness_model': max(successful_models.items(), key=lambda x: x[1]['robustness_score'])[0],
                    'avg_mse': round(np.mean(mse_values), 6),
                    'avg_mae': round(np.mean(mae_values), 6),
                    'avg_r2': round(np.mean(r2_values), 4),
                    'avg_robustness': round(np.mean(robustness_values), 4),
                    'models_evaluated': len(successful_models)
                }

            return {
                'success': True,
                'comparison_results': comparison_results,
                'timestamp': pd.Timestamp.now().isoformat()
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Error en comparación de modelos: {str(e)}"
            }