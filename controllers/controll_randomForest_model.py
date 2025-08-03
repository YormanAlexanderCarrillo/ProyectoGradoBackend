from flask import jsonify, request
from Models.randomForestModel import GasLevelRandomForest


model = GasLevelRandomForest()
# model.load_data('./data/sensor_mina_data.csv')
# model.load_data('./data/datos_sensor_procesados.csv')
model.load_data('./data/complete_data_normal.csv')
# model.load_data('./data/sensor_mina_data_parts_per_million_2.csv')


if model.model is None:
    model.train_model()

#ya esta listo en routes
def predict_gas_level():
    """
    Predice el nivel de gas basado en parámetros de entrada.
    Adaptado para devolver el objeto completo con la predicción y análisis de confiabilidad.
    """
    data = request.get_json()

    try:
        # Ahora el método predict devuelve un diccionario más completo
        prediction_result = model.predict(
            temperatura=data.get("temperatura"),
            humedad=data.get("humedad"),
            tiempo_calibracion=data.get("tiempo_calibracion"),
            nivel_bateria=data.get("nivel_bateria")
        )

        return jsonify({
            "success": True,
            "prediction": prediction_result["predicted_gas_level"],
            "confidence_interval": prediction_result.get("confidence_interval"),
            "lower_bound": prediction_result.get("lower_bound"),
            "upper_bound": prediction_result.get("upper_bound"),
            "reliability": prediction_result.get("reliability_analysis")
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
#ya esta listo en routes
def get_analysis_basic_stats():
    """
    Obtiene estadísticas básicas del conjunto de datos.
    Formato de respuesta sin cambios significativos.
    """
    basic_stats = model.get_basic_stats()

    return jsonify({
        "success": True,
        "basic_stats": basic_stats
    })
#ya esta listo en routes
def get_analysis_outliers():
    """
    Obtiene información sobre valores atípicos en el conjunto de datos..
    """
    outliers = model.detect_outliers([
        "temperatura_sensor",
        "humedad_ambiente",
        "nivel_gas_metano",
        "nivel_bateria"
    ])

    return jsonify({
        "success": True,
        "outliers": outliers
    })
#Ya esta listo en routes
def get_analysis_temporal_analysis():
    """
    Analiza la degradación temporal del sensor.
    Adaptado para incluir información adicional sobre la tendencia y degradación.
    """
    temporal_analysis = model.analyze_temporal_degradation()

    return jsonify({
        "success": True,
        "temporal_analysis": temporal_analysis
    })
#ya esta listo en routes
def get_analysis_correlations():
    """
    Obtiene la matriz de correlación entre variables.
    Adaptado para incluir interpretaciones de correlaciones.
    """
    correlations = model.get_correlations()

    return jsonify({
        "success": True,
        "correlations": correlations,
    })
#Ya eta listo en routes
def get_model_metrics_metrics():
    """
    Obtiene métricas de rendimiento del modelo.
    Sin cambios significativos en el formato.
    """
    metrics = model.get_training_results()["metrics"]

    return jsonify({
        "success": True,
        "metrics": metrics
    })

#ya esta listo en routes
def get_model_metrics_feature_importance():
    """
    Obtiene la importancia de características del modelo.
    Adaptado para el nuevo formato donde 'feature_importance' puede tener 'importances'
    en lugar de 'coefficients' dependiendo del modelo.
    """
    feature_importance = model.get_training_results()["feature_importance"]

    # Adaptamos el nombre de la clave para asegurar compatibilidad
    result = {
        "variables": feature_importance.get("variables")
    }

    # El RandomForest usa 'importances' mientras que el modelo lineal usa 'coefficients'
    if "importances" in feature_importance:
        result["importance_values"] = feature_importance["importances"]
    elif "coefficients" in feature_importance:
        result["importance_values"] = feature_importance["coefficients"]

    return jsonify({
        "success": True,
        "feature_importance": result
    })
#Ya esta listo en routes
def get_model_metrics_prediction_data():
    """
    Obtiene datos de predicción del modelo.
    Sin cambios significativos en el formato.
    """
    prediction_data = model.get_training_results()["prediction_data"]

    return jsonify({
        "success": True,
        "prediction_data": prediction_data
    })

#Ya esta listo en routes
def get_model_metrics_residuals():
    """
    Obtiene los residuos del modelo.
    Adaptado para incluir información adicional sobre normalidad.
    """
    training_results = model.get_training_results()
    residuals = training_results["residuals"]

    # Incluimos información sobre normalidad de residuos si está disponible
    residual_normality = training_results.get("residual_normality", {})

    return jsonify({
        "success": True,
        "residuals": residuals,
        "residual_normality": residual_normality
    })
#Ya esta listo en routes
def predict_data_gas_future():
    """
    Predice niveles futuros de gas.
    Adaptado para aceptar parámetros opcionales de inicio y devolver información adicional.
    """
    try:
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
            hours_ahead=hours,
            temperatura=temperatura,
            humedad=humedad,
            tiempo_calibracion=tiempo_calibracion,
            nivel_bateria=nivel_bateria

        )

        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

#No esta en routes --- -----
def impute_missing_values():
    """
    Imputa valores faltantes en los datos utilizando el método seleccionado.
    """
    data = request.get_json()

    try:
        columns = data.get("columns", ["temperatura_sensor", "humedad_ambiente", "nivel_gas_metano", "nivel_bateria"])
        method = data.get("method", "median")

        imputations = model.impute_missing_values(
            columns=columns,
            method=method
        )

        return jsonify({
            "success": True,
            "imputations": imputations
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

#No esta en routes --- ------
def correct_outliers():
    """
    Corrige valores atípicos en los datos utilizando el método seleccionado.
    """
    data = request.get_json()

    try:
        columns = data.get("columns", ["temperatura_sensor", "humedad_ambiente", "nivel_gas_metano", "nivel_bateria"])
        threshold = data.get("threshold", 3)
        method = data.get("method", "median")

        corrections = model.correct_outliers(
            columns=columns,
            threshold=threshold,
            method=method
        )

        return jsonify({
            "success": True,
            "corrections": corrections
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

#No esta en routes --- ------
def get_analysis_battery_impact():
    """
    Analiza el impacto del nivel de batería en las mediciones de gas.
    """
    try:
        battery_analysis = model.analyze_battery_impact()

        return jsonify({
            "success": True,
            "battery_analysis": battery_analysis
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

#No esta en routes --- ------
def get_analysis_temperature_impact():
    """
    Analiza el impacto de la temperatura en las mediciones de gas.
    """
    try:
        temperature_analysis = model.analyze_temperature_impact()

        return jsonify({
            "success": True,
            "temperature_analysis": temperature_analysis
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

#No esta en routes --- ----
def get_model_metrics_preprocessing():
    """
    Obtiene los resultados del preprocesamiento de datos durante el entrenamiento.
    """
    try:
        preprocessing = model.get_training_results().get("preprocessing", {})

        return jsonify({
            "success": True,
            "preprocessing": preprocessing
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# Este metodo puede ser ser implementado a futuro
def retrain_model(data_csv):
    """
    Fuerza el reentrenamiento del modelo con los datos actuales.
    """
    try:

        model.load_data(data_csv)

        training_results = model.train_model(force_retrain=True)

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