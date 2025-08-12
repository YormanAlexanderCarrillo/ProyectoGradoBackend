from flask import Blueprint
from controllers.controll_global_setting import (
    retrain_models_all
)

bp = Blueprint("setting", __name__)


@bp.route('/retrain', methods=['POST'])
def retrain_models():
    return retrain_models_all()
