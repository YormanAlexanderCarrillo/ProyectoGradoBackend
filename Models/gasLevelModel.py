import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
df = pd.read_csv('../data/sensor_mina_data.csv')

# Convertir fecha y hora a datetime
df['datetime'] = pd.to_datetime(df['fecha'] + ' ' + df['hora'])

# Análisis exploratorio inicial
print("Estadísticas descriptivas:")
print(df.describe())

print("\nCorrelaciones entre variables:")
correlations = df[['temperatura_sensor', 'humedad_ambiente',
                  'tiempo_desde_calibracion', 'nivel_gas_metano',
                  'nivel_bateria']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.tight_layout()
plt.savefig('correlaciones.png')
plt.close()

# Preparar los datos
X = df[['temperatura_sensor', 'humedad_ambiente',
        'tiempo_desde_calibracion', 'nivel_bateria']]
y = df['nivel_gas_metano']

# Escalar las variables para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMétricas de evaluación del modelo:")
print(f"Error cuadrático medio (MSE): {mse:.4f}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.4f}")
print(f"Error absoluto medio (MAE): {mae:.4f}")
print(f"R-cuadrado (R²): {r2:.4f}")

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)
plt.xlabel("Nivel real de gas metano (%)")
plt.ylabel("Nivel predicho de gas metano (%)")
plt.title("Predicciones vs Valores Reales")
plt.tight_layout()
plt.savefig('predicciones_metano.png')
plt.close()

# Análisis de la importancia de las variables
coef_df = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': model.coef_,
    'Coeficiente_abs': abs(model.coef_)
})
coef_df = coef_df.sort_values('Coeficiente_abs', ascending=False)

print("\nImportancia de las variables:")
print(coef_df[['Variable', 'Coeficiente']])

# Función para hacer predicciones con nuevos datos
def predecir_metano(temperatura, humedad, tiempo_calibracion, nivel_bateria):
    # Escalar los nuevos datos usando el mismo scaler
    nuevos_datos = np.array([[temperatura, humedad,
                             tiempo_calibracion, nivel_bateria]])
    nuevos_datos_scaled = scaler.transform(nuevos_datos)
    return model.predict(nuevos_datos_scaled)[0]

# Ejemplo de predicción
ejemplo = predecir_metano(
    temperatura=25,
    humedad=75,
    tiempo_calibracion=100,
    nivel_bateria=80
)
print(f"\nEjemplo de predicción:")
print(f"Para temperatura=25°C, humedad=75%, tiempo_calibracion=100h, "
      f"nivel_bateria=80%")
print(f"Nivel de metano predicho: {ejemplo:.2f}%")

# Análisis de residuos
residuos = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuos, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicciones")
plt.ylabel("Residuos")
plt.title("Gráfico de Residuos")
plt.tight_layout()
plt.savefig('residuos.png')
plt.close()