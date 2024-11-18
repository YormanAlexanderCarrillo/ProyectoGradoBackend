from flask import jsonify, request
from Models.randomForestModel import GasLevelRandomForest


model = GasLevelRandomForest()
model.load_data('./data/sensor_mina_data.csv')
model.train_model()


def predict_gas_level():
    data = request.get_json()

    try:

        predictionData = model.predict(
            temperatura=data.get("temperatura"),
            humedad=data.get("humedad"),
            tiempo_calibracion=data.get("tiempo_calibracion"),
            nivel_bateria=data.get("nivel_bateria")
        )

        #     'prediction': mean_prediction,
        #     'confidence_interval': confidence_interval,
        #     'lower_bound': mean_prediction - confidence_interval,
        #     'upper_bound': mean_prediction + confidence_interval

        prediction = predictionData["prediction"]

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
    training_results = model.train_model()

    metrics = training_results["metrics"]

    return jsonify({
        "success": True,
        "metrics": metrics
    })

def get_model_metrics_feature_importance():
    training_results = model.train_model()

    feature_importance = training_results["feature_importance"]

    return jsonify({
        "success": True,
        "feature_importance": feature_importance
    })


def get_model_metrics_prediction_data():

    training_results = model.train_model()

    prediction_data = training_results["prediction_data"]

    return jsonify({
        "success": True,
        "prediction_data": prediction_data
    })

def get_model_metrics_residuals():
    training_results = model.train_model()

    residuals = training_results["residuals"]

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