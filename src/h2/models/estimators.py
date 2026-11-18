"""Original fixed estimator configurations; no model selection."""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def model_instance(name):
    if name == "OLS":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearRegression(),
        )

    if name == "Ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        )

    if name == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=140,
            learning_rate=0.035,
            num_leaves=15,
            min_child_samples=150,
            reg_lambda=10.0,
            random_state=2026,
            n_jobs=2,
            verbosity=-1,
        )

    if name == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=140,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=150,
            subsample=1.0,
            colsample_bytree=0.85,
            reg_lambda=10.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=2026,
            n_jobs=2,
            verbosity=0,
        )

    raise KeyError(name)
