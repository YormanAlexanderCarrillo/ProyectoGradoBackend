from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from Models.gasLevelModel import (predecir_metano, detectar_atipicos)

app = Flask(__name__)
CORS(app)
df = pd.read_csv('./data/sensor_mina_data.csv')

@app.route('/', methods=['POST'])  # Cambiado a POST para recibir datos en JSON
def predict_gas_level():
    data = request.get_json()

    # Extraer los valores del JSON
    temperatura = data.get("temperatura")
    humedad = data.get("humedad")
    tiempo_calibracion = data.get("tiempo_calibracion")
    nivel_bateria = data.get("nivel_bateria")

    # Realizar la predicción
    ejemplo_pred = predecir_metano(temperatura, humedad, tiempo_calibracion, nivel_bateria)


    #datos atipicos
    # columnas_analizar = ['temperatura_sensor', 'humedad_ambiente', 'nivel_gas_metano', 'nivel_bateria']
    # atipicos = detectar_atipicos(df, columnas_analizar)
    #

    # Retornar la respuesta en formato JSON
    return jsonify({
        "temperatura": temperatura,
        "humedad": humedad,
        "tiempo_calibracion": tiempo_calibracion,
        "nivel_bateria": nivel_bateria,
        "prediccion_metano": round(ejemplo_pred, 2),
        # "atipicos": atipicos
    })

@app.route('/test', methods=['GET'])  # Cambiado a POST para recibir datos en JSON
def atipicos():


    #datos atipicos
    columnas_analizar = ['temperatura_sensor', 'humedad_ambiente', 'nivel_gas_metano', 'nivel_bateria']
    atipicos = detectar_atipicos(df, columnas_analizar)

    print(atipicos)

    # Retornar la respuesta en formato JSON
    return jsonify('hola')


@app.route('/datos_graficos', methods=['GET'])
def datos_graficos():
    # Datos para las gráficas
    print('entro metodo')
    correlaciones = df[['temperatura_sensor', 'humedad_ambiente', 'tiempo_desde_calibracion',
                        'nivel_gas_metano', 'nivel_bateria']].corr()

    # degradacion_temporal = df.groupby('dias_desde_calibracion')['nivel_gas_metano'].std().reset_index().to_dict(
    #     orient='records')
    json_correlaciones = correlaciones.to_json(orient='records')

    # Aquí incluyes más datos para las gráficas necesarias
    return jsonify({
        "correlaciones": json_correlaciones,
        # Añadir otros datos de gráficas aquí...
    })




if __name__ == '__main__':
    app.run()
