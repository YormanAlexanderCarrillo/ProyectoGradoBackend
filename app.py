from flask import Flask, request
from flask_cors import CORS
from Models.gasLevelModel import predecir_metano

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def hello_world():
    data = request.get_json()
    print(data["temperatura"])
    temperatura = data["temperatura"]
    humedad = data["humedad"]
    tiempo_calibracion = data["tiempo_calibracion"]
    ejemplo_pred = predecir_metano(temperatura, humedad, tiempo_calibracion)
    return (f"\nPredicción de nivel de metano para Temp={temperatura}°C, "
            f"Humedad={humedad}%, Tiempo desde calibración={tiempo_calibracion}h: "
            f"{ejemplo_pred:.2f} ppm")


if __name__ == '__main__':
    app.run()
