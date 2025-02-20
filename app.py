# app.py
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from routes.route_regresion_model import bp as regresion_bp
from routes.route_randomForest_model import bp as random_forest_bp
from routes.route_gradient_boosting_model import bp as gradient_boosting_bp


app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)

# Configuración de Swagger
SWAGGER_URL = '/api/doc'  # URL para accerder a la documentacion
API_URL = '/static/swagger.json'  # Ruta de los endpoint documentados

# Crear blueprint de Swagger UI
swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "API Regresión"
    }
)

# Registrar blueprints
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

app.register_blueprint(regresion_bp, url_prefix='/regresion')
app.register_blueprint(random_forest_bp, url_prefix="/random_forest")
app.register_blueprint(gradient_boosting_bp, url_prefix="/gradient_boosting")

# Ruta para servir el archivo swagger.json
@app.route('/static/swagger.json')
def serve_swagger_spec():
    return send_from_directory('config', 'swagger.json')

if __name__ == '__main__':
    app.run(debug=True)