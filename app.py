from flask import Flask, request, jsonify, blueprints
from flask_cors import CORS
from routes.route_regresion_model import bp as regresion_bp

app = Flask(__name__)
CORS(app, origins="http://localhost:5173", supports_credentials=True)


# Inicializar y entrenar el modelo al inicio

app.register_blueprint(regresion_bp, url_prefix='/regresion')




if __name__ == '__main__':
    app.run(debug=True)