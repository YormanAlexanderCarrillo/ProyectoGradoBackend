from flask import  jsonify, request
from Models.gasLevelModel import GasLevelModel


model = GasLevelModel()
# model.load_data('./data/sensor_mina_data_parts_per_million.csv')
# model.load_data('./data/sensor_mina_data.csv')
# model.load_data('./data/datos_sensor_procesados.csv')
model.load_data('./data/complete_data_normal.csv')


if model.model is None:
    model.train_model()


# predecir nivel de gas
def predict_gas_level():
    data = request.get_json()

    print(data)

    # El modelo esta respondiendo en un rango de 0 a 1
    # 0 es ausencia de gas
    # 1 Es concentración máxima de gas
    try:
        prediction_result = model.predict(
            temperatura=data.get("temperatura"),
            humedad=data.get("humedad"),
            tiempo_calibracion=data.get("tiempo_calibracion"),
            nivel_bateria=data.get("nivel_bateria")
        )

        print(prediction_result)

        return jsonify({
            "success": True,
            "prediction": prediction_result['predicted_gas_level'],
            "reliability": prediction_result['reliability_analysis']
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 40

#predecir nivel de gas futuro por numero de horas
def predict_data_gas_future():

    try:
        print("Prueba para predecir los niveles de gas a futuro")
        data = request.get_json()

        # Obtener número de horas y parámetros iniciales
        hours = data.get("hours")
        temperatura = data.get("temperatura")
        humedad = data.get("humedad")
        tiempo_calibracion = data.get("tiempo_calibracion")
        nivel_bateria = data.get("nivel_bateria")

        print(f"Horas a predecir: {hours}")
        print(
            f"Parámetros iniciales: Temp={temperatura}, Hum={humedad}, TCal={tiempo_calibracion}, Bat={nivel_bateria}")

        prediction = model.predict_gas_future(
            hours_ahead= hours,
            temperatura= temperatura,
            humedad = humedad,
            tiempo_calibracion= tiempo_calibracion,
            nivel_bateria= nivel_bateria

        )

        # prediction = model.predict_gas_future(hours_ahead=hours)

        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Analysis inicio
def get_analysis_basic_stats():
    # Obtener estadísticas básicas
    # Mostrar resúmenes estadísticos de todas las variables (media, desviación estándar, mínimo, máximo)
    # Crear visualizaciones de distribución de datos
    # Identificar rangos normales de operación para cada sensor
    basic_stats = model.get_basic_stats()

    print("Estadisticas basicas",basic_stats)

    return jsonify({
        "success": True,
        "basic_stats": basic_stats,
    })

# Determinar los datos atipicos
def get_analysis_outliers():
    # Detectar valores atípicos
    # Identificar mediciones anormales que podrían indicar fallos en los sensores
    # Visualizar puntos específicos donde los sensores pueden estar funcionando incorrectamente
    # Crear alertas para cuando se detecten valores fuera de rango
    outliers = model.detect_outliers([
        'temperatura_sensor',
        'humedad_ambiente',
        'nivel_gas_metano',
        'nivel_bateria'
    ])

    return jsonify({
        "success": True,
        "outliers": outliers,
    })


# Corregir los datos atipicos
def correct_data_outliers():
    # Corregir valores atípicos con valores predeterminados
    # Similar a cómo funciona get_analysis_outliers
    corrections = model.correct_outliers([
        'temperatura_sensor',
        'humedad_ambiente',
        'nivel_gas_metano',
        'nivel_bateria'
    ], threshold=3, method='mean')

    return jsonify({
        "success": True,
        "corrections": corrections,
    })

# Analisis temporal
def get_analysis_temporal_analysis():
    # Análisis de degradación temporal
    # Visualizar cómo la precisión del sensor se degrada con el tiempo
    # Identificar patrones en la variabilidad de las mediciones
    # Determinar cuándo es necesario recalibrar los sensores
    temporal_analysis = model.analyze_temporal_degradation()

    return jsonify({
        "success": True,
        "temporal_analysis": temporal_analysis,
    })

# Determina las correlaciones
def get_analysis_correlations():
    # Obtener correlaciones
    # Funciones de esta variable
    # Entender las relaciones entre diferentes variables (temperatura, humedad, nivel de batería, etc.)
    # Identificar qué factores tienen mayor impacto en las mediciones
    # Detectar dependencias entre variables que podrían afectar la precisión
    correlations = model.get_correlations()

    return jsonify({
        "success": True,
        "correlations": correlations,
    })

#  Analysis fin



# metodos para ver el impacto de las variables sobre los niveles de gas

# Impacto de la bateria sobre las mediciones

def get_analysis_battery_impact():
    # Analiza el impacto del nivel de batería en las mediciones de gas
    # Identifica rangos de batería donde las mediciones son menos confiables
    # Determina la degradación de la precisión con niveles bajos de batería
    battery_analysis = model.analyze_battery_impact()

    return jsonify({
        "success": True,
        "battery_analysis": battery_analysis,
    })

# Impacto de la temperatura sobre las mediciones
def get_analysis_temperature_impact():
    # Analiza el impacto de la temperatura del sensor en las mediciones de gas
    # Identifica rangos de temperatura donde las mediciones pueden ser inexactas
    # Determina la sensibilidad del sensor a cambios de temperatura
    temperature_analysis = model.analyze_temperature_impact()

    return jsonify({
        "success": True,
        "temperature_analysis": temperature_analysis,
    })


# model metrics inicio
def get_model_metrics_metrics():
    # Muestra cuatro tipo de errores (MSE (Error Cuadrático Medio), RMSE (Raíz del Error Cuadrático Medio), MAE (Error Absoluto Medio), R² (Coeficiente de Determinaci)
    # Con MSE y RMSE: Puedes determinar qué tan grandes son los errores de predicción en términos absolutos
    # Con MAE: Puedes explicar el error promedio a usuarios no técnicos de manera más intuitiva
    # Con R²: Puedes determinar qué porcentaje de la variabilidad en los datos explica tu modelo
    training_results = model.train_model()

    metrics = training_results["metrics"]

    return jsonify({
        "success": True,
        "metrics": metrics
    })

def get_model_metrics_feature_importance():
    #Importancia de Variables
    #Visualiza el impacto relativo de cada variable en las predicciones
    #Ayuda a identificar qué factores son más importantes en el modelo: temperatura_sensor, humedad_ambiente, tiempo_desde_calibracion, nivel_bateria
    #Determinar qué sensores son críticos para el mantenimiento
    training_results = model.train_model()

    feature_importance = training_results["feature_importance"]

    return jsonify({
        "success": True,
        "feature_importance": feature_importance
    })

def get_model_metrics_prediction_data():
    #Gráfico de Dispersión de Predicciones vs Valores Reales
    #Muestra qué tan bien se ajustan las predicciones a los valores reales
    #Detectar si hay valores atípicos que afectan al rendimiento
    #Determinar si el modelo tiene tendencia a sobre-predecir o sub-predecir

    training_results = model.train_model()

    prediction_data = training_results["prediction_data"]

    return jsonify({
        "success": True,
        "prediction_data": prediction_data
    })

def get_model_metrics_residuals():
    # Análisis de Residuos
    # Muestra la distribución de los errores del modelo
    # Ayuda a identificar patrones o sesgos en las predicciones
    # Identificar si hay sesgos sistemáticos en las predicciones
    # Determinar si hay rangos específicos donde el modelo necesita mejoras
    # Evaluar si los errores son aleatorios o siguen algún patrón que se pueda corregir
    # Decidir si el modelo necesita reentrenamiento basado en la distribución de errores
    training_results = model.train_model()

    residuals = training_results["residuals"]

    return jsonify({
        "success": True,
        "residuals": residuals
    })

def retrain_model(data_csv):
    """
    Fuerza el reentrenamiento del modelo con los datos actuales.
    """
    try:

        model.load_data(data_csv)

        training_results = model.train_model()

        return jsonify({
            "success": True,
            "message": "Modelo reentrenado exitosamente",
            "metrics": training_results["metrics"]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# model metrics fin




