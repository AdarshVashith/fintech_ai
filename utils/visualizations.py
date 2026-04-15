import plotly.express as px
import plotly.graph_objects as go

def plot_roc_curve(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC curve (area = {roc_auc:0.3f})',
                    line=dict(color='darkorange', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='navy', width=2, dash='dash')))
    fig.update_layout(
        title='Receiver Operating Characteristic',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        plot_bgcolor='white'
    )
    return fig

def plot_confusion_matrix(cm, classes=['Non-Default', 'Default']):
    fig = px.imshow(cm, text_auto=True, 
                    labels=dict(x="Predicted Label", y="True Label"),
                    x=classes, y=classes,
                    color_continuous_scale='Blues')
    fig.update_layout(title="Confusion Matrix")
    return fig

def plot_feature_importance(df_importance, top_n=15):
    fig = px.bar(df_importance.head(top_n), x='importance', y='feature', orientation='h',
                 title=f"Top {top_n} Feature Importances",
                 color='importance', color_continuous_scale='Viridis')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig
