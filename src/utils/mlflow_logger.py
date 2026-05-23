import mlflow
import os


def setup_mlflow(cfg, model_name):
    experiment_name = cfg['experiment'].get('name', model_name)

    # local: saves to mlruns/ folder in project root
    # on Kaggle: saves to /kaggle/working/mlruns/
    tracking_uri = (
        '/kaggle/working/mlruns'
        if os.environ.get('KAGGLE_KERNEL_RUN_TYPE')
        else './mlruns'
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_params(cfg):
    mlflow.log_params({
        "model": cfg['model']['name'],
        "num_classes": cfg['model']['num_classes'],
        "batch_size": cfg['training']['batch_size'],
        "learning_rate": cfg['training']['learning_rate'],
        "weight_decay": cfg['training']['weight_decay'],
        "epochs": cfg['training']['epochs'],
        "optimizer":  cfg['training']['optimizer'],
        "scheduler": cfg['training']['scheduler'],
        "seed": cfg['experiment']['seed'],
    })

def log_epoch(epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, lr):
    mlflow.log_metrics({
        "train/loss": train_loss,
        "train/acc": train_acc,
        "train/f1": train_f1,
        "val/loss": val_loss,
        "val/acc": val_acc,
        "val/f1": val_f1,
        "lr": lr,
    }, step=epoch)


def log_model(model, model_name):
    mlflow.pytorch.log_model(model, artifact_path=model_name)


def end_run():
    mlflow.end_run()