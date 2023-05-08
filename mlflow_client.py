import os
import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.pixelbrain.nvquynh.codes"
os.environ["MLFLOW_TRACKING_USERNAME"] = "mlflow"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "mlflow"

client = mlflow.MlflowClient()

def list_models(experiment_name):
  models = []
  runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string="attributes.status = \"FINISHED\"",
    run_view_type=ViewType.ACTIVE_ONLY,
    order_by=["metrics.val_accuracy DESC"]
  )
  
  for _, run in runs.iterrows():
    run_info = dict(run)
    print(run_info.get('tags.mlflow.runName'))
    # models.append({
    #   'model_url': f"gs://pixelbrain/models/{experiment_name}/{run_info.get('run_id')}",
    #   'experiment_id': run_info.get('experiment_id'),
    #   'run_name': run_info.get('tags.mlflow.runName'),
    #   'train_accuracy': run_info.get('metrics.train_accuracy'),
    #   'val_accuracy': run_info.get('metrics.val_accuracy'),
    #   'train_loss': run_info.get('metrics.train_loss'),
    #   'val_loss': run_info.get('metrics.val_loss'),
    # })
  return models

def find_argmax_by_attr(lst, attr):
    max_val = None
    max_obj = None
    max_idx = None
    for i, obj in enumerate(lst):
        val = getattr(obj, attr)
        if max_val is None or val > max_val:
            max_val = val
            max_obj = obj
            max_idx = i
    return max_idx, max_val, max_obj

def parse_training_history(history):
  data = []
  for idx, metric in enumerate(history):
    name = f"{idx}"
    value = round(metric.value, 2)
    data.append({ 'name': name, 'value': value })
  return data


def get_training_graph(run_id):
  val_acc_history = client.get_metric_history(run_id, "val_accuracy")
  val_loss_history = client.get_metric_history(run_id, "val_loss")
  data = {
    'val_acc_history': parse_training_history(val_acc_history),
    'val_loss_history': parse_training_history(val_loss_history),
  }
  return data

def best_accuracy(experiment_name, for_deployment=False):
  best_val_accuracy = 0
  best_run_info = {}
  filter_string = ""
  if for_deployment:
    print('Getting best model for deployment')
    filter_string = "attributes.status = \"FINISHED\""
  runs = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string=filter_string,
    run_view_type=ViewType.ACTIVE_ONLY,
    order_by=["metrics.val_accuracy DESC"]
  )

  if len(runs) == 0:
    return

  for _, run in runs.iterrows():
    run_info = dict(run)
    val_accuracy_history = client.get_metric_history(run_info.get('run_id'), "val_accuracy")
    max_idx, max_value, max_metric = find_argmax_by_attr(val_accuracy_history, 'value')
    if max_value and max_value > best_val_accuracy:
      best_val_accuracy = max_value
      best_run_info = run_info
  
  result = { 
    'best_val_accuracy': round(best_val_accuracy * 100, 2), 
    'run_id': best_run_info.get('run_id'),
    'run_name': best_run_info.get('tags.mlflow.runName'),
    'model_url': f"gs://pixelbrain/models/{experiment_name}/{best_run_info.get('run_id')}",
  }

  return result

