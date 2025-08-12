from flask import jsonify, request
import os
from werkzeug.utils import secure_filename
from datetime import datetime

from controllers.controll_prophet_model import (
    retrain_model as retrain_prophet
)

from controllers.controll_gradient_boosting_model import (
    retrain_model as retrain_boosting
)

from controllers.controll_randomForest_model import (
    retrain_model as retrain_random
)

from controllers.controll_Regrion_Model import retrain_model as retrain_regresion

def retrain_models_all():
    """
    Metodo para reentrenar todos los modelos
    """

    if 'file' not in request.files:
        return jsonify({
            "status": False,
            "error": "No se envió ningún archivo"
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            "status": False,
            "error": "No se seleccionó ningún archivo"
        }), 400

        # Verificar extensión del archivo (opcional)
    if not file.filename.lower().endswith('.csv'):
        return jsonify({
            "status": False,
            "error": "Solo se permiten archivos CSV"
        }), 400

        # Crear nombre único con timestamp
    original_filename = secure_filename(file.filename)
    name, ext = os.path.splitext(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{name}_{timestamp}{ext}"

    # Crear directorio data si no existe
    data_folder = './data'
    os.makedirs(data_folder, exist_ok=True)

    file_path = os.path.join(data_folder, unique_filename)

    file.save(file_path)

    retrain_boosting(file_path)
    retrain_random(file_path)
    retrain_prophet(file_path)
    retrain_regresion(file_path)

    try:

        return jsonify({
            "status": True,
            "message": "Modelos reentrandos con exito"
        })
    except Exception as e:
        return jsonify({
            "status": False,
            "error": str(e)
        }), 500



