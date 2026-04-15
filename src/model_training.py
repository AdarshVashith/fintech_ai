from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    print(f"XGBoost not available: {e}")
    HAS_XGB = False

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_logistic_regression(X_train, y_train):
    param_grid = {'classifier__C': [0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=3)
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    grid = GridSearchCV(lr_pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(grid.best_estimator_, 'models/logistic_regression.pkl')
    return grid.best_estimator_

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/random_forest.pkl')
    return rf

def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(dt, 'models/decision_tree.pkl')
    return dt

def train_xgboost(X_train, y_train):
    if not HAS_XGB:
        print("Skipping XGBoost due to missing library (e.g. libomp).")
        return None
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb, 'models/xgboost.pkl')
    return xgb
