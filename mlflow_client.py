import os
import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.pixelbrain.nvquynh.codes"
os.environ["MLFLOW_TRACKING_USERNAME"] = "mlflow"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "mlflow"

def list_models(experiment_name):
  models = []
  runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string="",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=10,
    order_by=["metrics.val_accuracy DESC"]
  )
  
  for _, run in runs.iterrows():
    run_info = dict(run)
    models.append({
      'model_url': f"gs://pixelbrain/models/{experiment_name}/{run_info.get('run_id')}",
      'experiment_id': run_info.get('experiment_id'),
      'run_name': run_info.get('tags.mlflow.runName'),
      'train_accuracy': run_info.get('metrics.train_accuracy'),
      'val_accuracy': run_info.get('metrics.val_accuracy'),
      'train_loss': run_info.get('metrics.train_loss'),
      'val_loss': run_info.get('metrics.val_loss'),
    })
  return models

def best_model(experiment_name):
  return list_models(experiment_name)[0]

# print(best_model('z6c22med1p'))