from flask import Blueprint

import controllers.controll_Regrion_Model
from controllers.controll_Regrion_Model import get_analysis_basic_stats, get_analysis_outliers, \
    get_analysis_temporal_analysis, get_analysis_correlations, predict_gas_level, get_model_metrics_metrics, \
    get_model_metrics_feature_importance, get_model_metrics_residuals, get_model_metrics_prediction_data, \
    predict_data_gas_future, correct_data_outliers, get_analysis_battery_impact, predict_reading_uncertainty, \
    get_error_analysis_summary, predict_calibration_error, get_battery_gas_real_data
bp = Blueprint('regresion', __name__)


@bp.route('/analysis/basic_stats', methods=['GET'])
def basic_stats():
    return get_analysis_basic_stats()


@bp.route('/analysis/outliers', methods=['GET'])
def outliers():
    return get_analysis_outliers()


@bp.route('/analysis/correct-outliers', methods=['GET'])
def correctOutliers():
    return correct_data_outliers()


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


# Nuevos metodos -----------------

@bp.route('/analysis/battery-impact', methods=['GET'])
def impact_battery():
    return get_analysis_battery_impact()


@bp.route('/analysis/temperature_impact', methods=['GET'])
def impact_temperature():
    return controllers.controll_Regrion_Model.get_analysis_temperature_impact();


# -------------------------
@bp.route('/predict/prediction_future', methods=['POST'])
def prediction_gas_future():
    return predict_data_gas_future()


@bp.route('/predict/calibration_error', methods=['POST'])
def predict_calibration_error_route():
    return predict_calibration_error()


@bp.route('/predict/reading_uncertainty', methods=['POST'])
def predict_reading_uncertainty_route():
    return predict_reading_uncertainty()


@bp.route('/analysis/error_summary', methods=['GET'])
def error_analysis_summary():
    return get_error_analysis_summary()


@bp.route('/data/battery-gas', methods=['GET'])
def battery_gas_data():
    return get_battery_gas_real_data()

