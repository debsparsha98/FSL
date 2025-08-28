import os
import yaml
from super_gradients.training import Trainer, models
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095

# Constants for file paths and naming
ROOT = "D:/project_root"
CONFIG_PATH = os.path.join(ROOT, "data", "data_config.yaml")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "checkpoints")
EXPERIMENT_NAME = "yolo_nas_exp"

# No-op callback to prevent NoneType callable errors in detection metrics
def noop_post_prediction_callback(preds, device=None):
    return preds

def main():
    # Load dataset parameters from YAML
    with open(CONFIG_PATH, "r") as f:
        data_params = yaml.safe_load(f)

    # Create training and validation dataloaders in COCO-YOLO format
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": os.path.join(ROOT, "data"),
            "images_dir": "images/train",
            "labels_dir": "labels/train",
            "classes": data_params["class_names"],
        },
        dataloader_params={
            "batch_size": 2,
            "num_workers": 2,
            "shuffle": True,
            "pin_memory": True,
        },
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": os.path.join(ROOT, "data"),
            "images_dir": "images/val",
            "labels_dir": "labels/val",
            "classes": data_params["class_names"],
        },
        dataloader_params={
            "batch_size": 2,
            "num_workers": 2,
            "shuffle": False,
            "pin_memory": True,
        },
    )

    # Initialize YOLO NAS small model without pretrained weights
    model = models.get(
        "yolo_nas_s",
        num_classes=len(data_params["class_names"]),
        pretrained_weights=None,
    )

    # Initialize Trainer instance
    trainer = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=OUTPUT_DIR)

    # Define PPYOLOE loss
    loss = PPYoloELoss(num_classes=len(data_params["class_names"]))

    # Define metrics with no-op callback to avoid errors
    metrics = [
        DetectionMetrics_050(
            num_cls=len(data_params["class_names"]),
            post_prediction_callback=noop_post_prediction_callback,
        ),
        DetectionMetrics_050_095(
            num_cls=len(data_params["class_names"]),
            post_prediction_callback=noop_post_prediction_callback,
        ),
    ]

    # Define training parameters
    train_params = {
        "max_epochs": 50,
        "average_best_models": True,
        "warmup_epochs": 3,
        "initial_lr": 0.001,
        "optimizer": "SGD",
        "loss": loss,
        "train_metrics_list": metrics,
        "valid_metrics_list": metrics,
        "metric_to_watch": "mAP@0.50",
        "greater_metric_to_watch_is_better": True,
        "mixed_precision": True,
    }

    # Start training
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data,
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Required for Windows multiprocess support
    main()
