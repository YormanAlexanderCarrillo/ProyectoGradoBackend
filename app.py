from flask import Flask, request, jsonify
from flask_cors import CORS
from Models.gasLevelModel import GasLevelModel

app = Flask(__name__)
CORS(app, origins="http://localhost:5173", supports_credentials=True)


# Inicializar y entrenar el modelo al inicio
model = GasLevelModel()
model.load_data('./data/sensor_mina_data.csv')
model.train_model()


@app.route('/predict', methods=['POST'])
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


@app.route('/analysis', methods=['GET'])
def get_analysis():
    # Obtener estadísticas básicas
    basic_stats = model.get_basic_stats()

    # Detectar valores atípicos
    outliers = model.detect_outliers([
        'temperatura_sensor',
        'humedad_ambiente',
        'nivel_gas_metano',
        'nivel_bateria'
    ])

    # Análisis de degradación temporal
    temporal_analysis = model.analyze_temporal_degradation()

    # Obtener correlaciones
    correlations = model.get_correlations()

    return jsonify({
        "success": True,
        "basic_stats": basic_stats,
        "outliers": outliers,
        "temporal_analysis": temporal_analysis,
        "correlations": correlations
    })


@app.route('/model-metrics', methods=['GET'])
def get_model_metrics():
    # Re-entrenar el modelo y obtener métricas
    training_results = model.train_model()

    return jsonify({
        "success": True,
        "results": training_results
    })


if __name__ == '__main__':
    app.run(debug=True)