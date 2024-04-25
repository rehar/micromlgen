from micromlgen.utils import jinja, check_type


def is_gradientboosting_regressor(clf):
    """
    Test if classifier can be ported
    """
    return check_type(clf, 'GradientBoostingRegressor')


def port_gradientboosting_regressor(clf, **kwargs):
    """
    Port sklearn's GradientBoostingRegressor
    """
    return jinja('gradientboosting/gradientboosting_regressor.jinja', {
        'dtype': 'float',
        'n_estimators': clf.n_estimators,
        'trees': [{
            'left': clf.tree_.children_left,
            'right': clf.tree_.children_right,
            'features': clf.tree_.feature,
            'thresholds': clf.tree_.threshold,
            'values': clf.tree_.value,
        } for clf in clf.estimators_[1:clf.estimators_.size, 0]]
    }, {
        'classname': 'GradientBoostingRegressor'
    }, **kwargs)
