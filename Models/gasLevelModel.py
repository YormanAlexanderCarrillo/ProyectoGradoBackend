import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

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

    def load_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['datetime'] = pd.to_datetime(self.df['fecha'] + ' ' + self.df['hora'])
        self.df['dias_desde_calibracion'] = self.df['tiempo_desde_calibracion'] / 24
        # el metodo get_basic_stats() retorna las estadísticas descriptivas del conjunto de datos.
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