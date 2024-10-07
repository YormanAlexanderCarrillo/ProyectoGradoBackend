import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('./data/sensor_mina_data.csv')

# Preparar los datos
X = df[['temperatura_sensor', 'humedad_ambiente', 'tiempo_desde_calibracion']]
y = df['nivel_gas_metano']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error cuadrático medio: {mse:.2f}")
print(f"R-cuadrado: {r2:.2f}")

# Visualizar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Nivel real de gas metano")
plt.ylabel("Nivel predicho de gas metano")
plt.title("Predicciones vs Valores Reales")
plt.tight_layout()
plt.savefig('predicciones_metano.png')
plt.close()

# Imprimir los coeficientes del modelo
coef_df = pd.DataFrame(list(zip(X.columns, model.coef_)), columns=['Variable', 'Coeficiente'])
print("\nCoeficientes del modelo:")
print(coef_df)


# Función para hacer predicciones
def predecir_metano(temperatura, humedad, tiempo_calibracion):
    return model.predict([[temperatura, humedad, tiempo_calibracion]])[0]

