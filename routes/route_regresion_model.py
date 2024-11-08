from flask import Blueprint
from controllers.controll_Regrion_Model import  get_analysis_basic_stats, get_analysis_outliers, get_analysis_temporal_analysis,get_analysis_correlations, predict_gas_level, get_model_metrics_metrics, get_model_metrics_feature_importance, get_model_metrics_residuals, get_model_metrics_prediction_data


bp = Blueprint('regresion', __name__)

@bp.route('/analysis/basic_stats', methods=['GET'])
def basic_stats():
    return get_analysis_basic_stats()

@bp.route('/analysis/outliers', methods=['GET'])
def outliers():
    return get_analysis_outliers()

@bp.route('/analysis/temporal_analysis', methods=['GET'])
def temporal_analysis():
    return get_analysis_temporal_analysis()

@bp.route('/analysis/correlations', methods=['GET'])
def correlations():
    return get_analysis_correlations()

@bp.route('/predict', methods=['POST'])
def predict_gas():
    return predict_gas_level()

@bp.route('/model_metrics/metrics', methods=['GET'])
def model_metrics():
    return get_model_metrics_metrics()

@bp.route('/model_metrics/feature_importance', methods=['GET'])
def feature_importance():
    return get_model_metrics_feature_importance()

@bp.route('/model_metrics/residuals', methods=['GET'])
def residuals():
    return get_model_metrics_residuals()

@bp.route('/model_metrics/prediction_data', methods=['GET'])
def prediction_data():
    return get_model_metrics_prediction_data()

