from flask import jsonify, request
from Models.gradient_boosting_model import GasLevelGradientBoostingModel


model = GasLevelGradientBoostingModel()
model.load_data('./data/sensor_mina_data.csv')


if model.model is None:
 training_results = model.train_model()

def predict_gas_level():
    data = request.get_json()

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


def get_analysis_basic_stats():
    basic_stats = model.get_basic_stats()

    return jsonify({
        "success": True,
        "basic_stats": basic_stats
    })

def get_analysis_outliers():
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

def get_analysis_temporal_analysis():
    temporal_analysis = model.analyze_temporal_degradation()

    return jsonify({
        "success": True,
        "temporal_analysis": temporal_analysis
    })

def get_analysis_correlations():
    correlations = model.get_correlations()

    return jsonify({
        "success": True,
        "correlations": correlations
    })

def get_model_metrics_metrics():
    metrics = model.get_training_results()["metrics"]

    return jsonify({
        "success": True,
        "metrics": metrics
    })

def get_model_metrics_feature_importance():

    feature_importance = model.get_training_results()["feature_importance"]


    return jsonify({
        "success": True,
        "feature_importance": feature_importance
    })


def get_model_metrics_prediction_data():
    prediction_data = model.get_training_results()["prediction_data"]


    return jsonify({
        "success": True,
        "prediction_data": prediction_data
    })

def get_model_metrics_residuals():
    residuals = model.get_training_results()["residuals"]


    return jsonify({
        "success": True,
        "residuals": residuals
    })

def predict_data_gas_future():
    try:
        data = request.get_json()

        hours = data.get("hours")
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