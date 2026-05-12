from flask import jsonify, request
from Models.prophet_model import GasLevelProphetModel
from Models.gasLevelModel import GasLevelModel


model = GasLevelProphetModel()
# model.load_data('./data/sensor_mina_data.csv')
# model.load_data('./data/datos_sensor_procesados.csv')
model.load_data('./data/complete_data_normal.csv')

if model.model is None:
    model.train_model()

def predict_gas_level():
    """
    Predice el nivel de gas basado en parámetros de entrada.
    """
    data = request.get_json()

    try:
        prediction_result = model.predict(
            temperatura=data.get("temperatura"),
            humedad=data.get("humedad"),
            tiempo_calibracion=data.get("tiempo_calibracion"),
            nivel_bateria=data.get("nivel_bateria")
        )

        # Prophet adicionalmente proporciona intervalos de confianza
        return jsonify({
            "success": True,
            "prediction": round(prediction_result["predicted_gas_level"], 2),
            "reliability_analysis": prediction_result["reliability_analysis"],
            "confidence_interval": prediction_result.get("confidence_interval", {})
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

def get_analysis_basic_stats():
    """
    Obtiene estadísticas básicas del conjunto de datos.
    """
    basic_stats = model.get_basic_stats()

    return jsonify({
        "success": True,
        "basic_stats": basic_stats
    })

def get_analysis_outliers():
    """
    Obtiene información sobre valores atípicos en el conjunto de datos.
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

def get_analysis_temporal_analysis():
    """
    Analiza la degradación temporal del sensor.
    """
    temporal_analysis = model.analyze_temporal_degradation()

    return jsonify({
        "success": True,
        "temporal_analysis": temporal_analysis
    })

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

def get_analysis_correlations():
    """
    Obtiene la matriz de correlación entre variables.
    """
    correlations = model.get_correlations()

    return jsonify({
        "success": True,
        "correlations": correlations,
    })

def get_model_metrics_metrics():
    """
    Obtiene métricas de rendimiento del modelo.
    """
    metrics = model.get_training_results()["metrics"]

    return jsonify({
        "success": True,
        "metrics": metrics,
    })

def get_model_metrics_feature_importance():
    """
    Obtiene la importancia de características del modelo.
    """
    feature_importance = model.get_training_results()["feature_importance"]

    # Adaptamos el nombre de la clave para asegurar compatibilidad
    result = {
        "variables": feature_importance.get("variables")
    }

    # El Prophet usa 'importances' igual que GradientBoosting
    if "importances" in feature_importance:
        result["importance_values"] = feature_importance["importances"]

    return jsonify({
        "success": True,
        "feature_importance": result
    })

def get_model_metrics_prediction_data():
    """
    Obtiene datos de predicción del modelo.
    """
    prediction_data = model.get_training_results()["prediction_data"]

    return jsonify({
        "success": True,
        "prediction_data": prediction_data
    })

def get_model_metrics_residuals():
    """
    Obtiene los residuos del modelo.
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

def get_model_prophet_components():
    """
    Obtiene los componentes específicos del modelo Prophet (tendencia, estacionalidad, etc.)
    """
    try:
        training_results = model.get_training_results()
        prophet_components = training_results.get("prophet_components", {})

        return jsonify({
            "success": True,
            "prophet_components": prophet_components
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

def retrain_model(data_csv):
    """
    Fuerza el reentrenamiento del modelo con los datos actuales.
    """
    try:
        # cargar datos enviados
        model.load_data(data_csv)

        training_results = model.train_model(force_retrain=True)

        return jsonify({
            "success": True,
            "message": "Modelo Prophet reentrenado exitosamente",
            "metrics": training_results["metrics"]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def predict_data_gas_future():
    """
    Predice niveles futuros de gas.
    """
    try:
        data = request.get_json()

        hours = data.get("hours", 24)
        temperatura = data.get("temperatura")
        humedad = data.get("humedad")
        tiempo_calibracion = data.get("tiempo_calibracion")
        nivel_bateria = data.get("nivel_bateria")

        # Si se proporcionan todos los parámetros iniciales, los usamos
        if all(param is not None for param in [temperatura, humedad, tiempo_calibracion, nivel_bateria]):
            prediction = model.predict_gas_future(
                hours_ahead=hours,
                temperatura=temperatura,
                humedad=humedad,
                tiempo_calibracion=tiempo_calibracion,
                nivel_bateria=nivel_bateria
            )
        else:
            # De lo contrario, usamos los últimos valores conocidos
            prediction = model.predict_gas_future(hours_ahead=hours)

        return jsonify({
            "success": True,
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def predict_calibration_error():
    """
    Predice el error de calibración del sensor basado en condiciones operacionales.
    """
    try:
        data = request.get_json()

        # Validar datos de entrada
        required_fields = ['temperatura', 'humedad', 'tiempo_calibracion', 'nivel_bateria']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Campo requerido faltante: {field}"
                }), 400

        # Realizar predicción de error de calibración
        error_result = model.predict_calibration_error(
            temperatura=data.get("temperatura"),
            humedad=data.get("humedad"),
            tiempo_calibracion=data.get("tiempo_calibracion"),
            nivel_bateria=data.get("nivel_bateria")
        )

        return jsonify({
            "success": True,
            "error_calibration_prediction": error_result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def predict_reading_uncertainty():
    """
    Predice la incertidumbre y confiabilidad de la lectura actual.
    """
    try:
        data = request.get_json()

        # Validar datos de entrada
        required_fields = ['temperatura', 'humedad', 'tiempo_calibracion', 'nivel_bateria']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Campo requerido faltante: {field}"
                }), 400

        # Realizar predicción de incertidumbre
        uncertainty_result = model.predict_reading_uncertainty(
            temperatura=data.get("temperatura"),
            humedad=data.get("humedad"),
            tiempo_calibracion=data.get("tiempo_calibracion"),
            nivel_bateria=data.get("nivel_bateria")
        )

        return jsonify({
            "success": True,
            "uncertainty_prediction": uncertainty_result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def get_error_analysis_summary():
    """
    Obtiene un resumen del análisis de errores y capacidades del modelo.
    """
    try:
        summary = {
            "modelo_tipo": "Prophet",
            "capacidades_error": {
                "prediccion_deriva_calibracion": True,
                "analisis_incertidumbre": True,
                "evaluacion_confiabilidad": True,
                "recomendaciones_mantenimiento": True
            },
            "factores_evaluados": {
                "deriva_temporal": "Degradación por tiempo desde calibración",
                "deriva_temperatura": "Efecto de temperatura en precisión",
                "deriva_humedad": "Efecto de humedad en mediciones",
                "deriva_bateria": "Impacto del nivel de batería"
            },
            "rangos_recomendados": {
                "temperatura_optima": "20-25°C",
                "humedad_optima": "45-75%",
                "tiempo_max_calibracion": "168 horas (7 días)",
                "nivel_min_bateria": "50%"
            },
            "interpretacion_severidad": {
                "bajo": "Error < 0.5% - Monitoreo rutinario",
                "moderado": "Error 0.5-1.0% - Verificación frecuente",
                "alto": "Error 1.0-2.0% - Calibración pronto",
                "critico": "Error > 2.0% - Calibración inmediata"
            }
        }

        return jsonify({
            "success": True,
            "error_analysis_summary": summary
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500