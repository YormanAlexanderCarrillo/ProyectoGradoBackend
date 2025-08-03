from flask import jsonify
from services.service_comparative_models import ComparativeModels
comparative_controller = ComparativeModels()


def get_models_comparison():
    """
    Endpoint principal para obtener la comparación de todos los modelos.

    Returns:
        JSON: Resultados comparativos de los modelos predictivos
    """
    try:
        result = comparative_controller.compare_all_models()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Error interno del servidor: {str(e)}"
        }), 500


def get_detailed_metrics():
    """
    Endpoint para obtener métricas detalladas de cada modelo individual.

    Returns:
        JSON: Métricas detalladas por modelo
    """
    try:
        detailed_results = {}

        for model_name, model_instance in comparative_controller.models.items():
            try:
                if hasattr(model_instance, 'get_model_metrics'):
                    metrics = model_instance.get_model_metrics()
                    detailed_results[model_name] = metrics
                else:
                    detailed_results[model_name] = {
                        'info': 'Métricas detalladas no disponibles para este modelo'
                    }
            except Exception as e:
                detailed_results[model_name] = {
                    'error': f"Error obteniendo métricas: {str(e)}"
                }

        return jsonify({
            'success': True,
            'detailed_metrics': detailed_results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Error obteniendo métricas detalladas: {str(e)}"
        }), 500