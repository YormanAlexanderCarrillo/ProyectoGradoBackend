from flask import  jsonify, request
from Models.gasLevelModel import GasLevelModel


model = GasLevelModel()
model.load_data('./data/sensor_mina_data.csv')
model.train_model()



def predict_gas_level():
    data = request.get_json()

    # El modelo esta respondiendo en un rango de 0 a 1
    # 0 es ausencia de gas
    # 1 Es concentración máxima de gas
    try:
        prediction = model.predict(
            temperatura=data.get("temperatura"),
            humedad=data.get("humedad"),
            tiempo_calibracion=data.get("tiempo_calibracion"),
            nivel_bateria=data.get("nivel_bateria")
        )

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# Analysis inicio
def get_analysis_basic_stats():
    # Obtener estadísticas básicas
    # Mostrar resúmenes estadísticos de todas las variables (media, desviación estándar, mínimo, máximo)
    # Crear visualizaciones de distribución de datos
    # Identificar rangos normales de operación para cada sensor
    basic_stats = model.get_basic_stats()

    return jsonify({
        "success": True,
        "basic_stats": basic_stats,
    })

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

    temporal_analysis = model.analyze_temporal_degradation()
    basic_stats = model.get_basic_stats()
    correlations = model.get_correlations()


    return jsonify({
        "success": True,
        "outliers": outliers,
    })

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

# model metrics inicio
def get_model_metrics_metrics():
    # Re-entrenar el modelo y obtener métricas
    training_results = model.train_model()

    metrics = training_results["metrics"]

    return jsonify({
        "success": True,
        "metrics": metrics
    })

def get_model_metrics_feature_importance():
    # Re-entrenar el modelo y obtener métricas
    training_results = model.train_model()

    feature_importance = training_results["feature_importance"]

    return jsonify({
        "success": True,
        "feature_importance": feature_importance
    })

def get_model_metrics_prediction_data():
    # Re-entrenar el modelo y obtener métricas
    training_results = model.train_model()

    prediction_data = training_results["prediction_data"]

    return jsonify({
        "success": True,
        "prediction_data": prediction_data
    })

def get_model_metrics_residuals():
    # Re-entrenar el modelo y obtener métricas
    training_results = model.train_model()

    residuals = training_results["residuals"]

    return jsonify({
        "success": True,
        "metrics": residuals
    })

