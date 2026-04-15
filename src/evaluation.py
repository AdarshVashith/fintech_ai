from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd

def evaluate_model(y_true, y_pred, y_pred_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
    }
    
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return metrics, cm, cr

def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    elif hasattr(model, 'coef_'):
        importances = model.coef_[0]
        return pd.DataFrame({'feature': feature_names, 'importance': abs(importances)}).sort_values('importance', ascending=False)
    return None
