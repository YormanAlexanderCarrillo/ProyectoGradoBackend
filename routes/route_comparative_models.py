from flask import Blueprint
from controllers.controll_comparative_model import get_models_comparison, get_detailed_metrics

bp = Blueprint('comparative_models', __name__)


@bp.route('/models_comparison', methods=['GET'])
def models_comparison():
    """
    Endpoint principal para comparación de modelos predictivos.

    Returns:
        JSON: Resultados comparativos incluyendo MSE, MAE, R², y robustez
    """
    return get_models_comparison()


@bp.route('/detailed_metrics', methods=['GET'])
def detailed_metrics():
    """
    Endpoint para métricas detalladas de cada modelo individual.

    Returns:
        JSON: Métricas específicas de cada modelo
    """
    return get_detailed_metrics()


@bp.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar el estado del módulo de pruebas comparativas.

    Returns:
        JSON: Estado del servicio
    """
    from flask import jsonify
    return jsonify({
        'status': 'active',
        'module': 'comparative_tests',
        'available_endpoints': [
            '/models_comparison',
            '/detailed_metrics',
            '/health'
        ]
    })