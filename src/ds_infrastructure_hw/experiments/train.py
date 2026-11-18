"""Run one reproducible CIFAR-10 training experiment."""

import argparse
import json
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ds_infrastructure_hw.experiments.dataset import CIFAR10S3Dataset
from ds_infrastructure_hw.experiments.metrics import (
    EvaluationResult,
    evaluate,
)
from ds_infrastructure_hw.experiments.models import build_model
from ds_infrastructure_hw.s3_client import get_bucket_name, get_s3_client


def read_config(path: Path) -> dict[str, Any]:
    """Read and validate a YAML experiment configuration."""
    with path.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)

    if not isinstance(value, dict):
        raise ValueError("Experiment configuration must be a mapping")

    return cast(dict[str, Any], value)


def get_git_commit() -> str:
    """Get the Git commit from environment or local repository."""
    import os

    commit = os.environ.get("GIT_COMMIT")

    if commit:
        return commit

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "unknown"

    if result.returncode != 0:
        return "unknown"

    return result.stdout.strip()


def flatten_config(
    values: dict[str, Any],
    prefix: str = "",
) -> dict[str, str]:
    """Convert nested YAML values into MLflow parameters."""
    result: dict[str, str] = {}

    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_config(cast(dict[str, Any], value), name))
        else:
            result[name] = json.dumps(value, ensure_ascii=False)

    return result


def save_confusion_matrix(
    confusion: list[list[int]],
    destination: Path,
) -> None:
    """Save a validation confusion matrix as a PNG image."""
    figure, axis = plt.subplots(figsize=(9, 8))

    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis)

    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Validation confusion matrix")

    axis.set_xticks(range(10))
    axis.set_yticks(range(10))

    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Perform one training epoch."""
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        optimizer.zero_grad()

        logits = cast(torch.Tensor, model(images))
        loss = cast(torch.Tensor, criterion(logits, labels))

        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot train on an empty dataset")

    return total_loss / total_samples


def upload_results(
    experiment_name: str,
    run_name: str,
    directory: Path,
) -> str:
    """Upload the best checkpoint and metadata to S3."""
    for value in (experiment_name, run_name):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError(f"Invalid experiment/run name: {value}")

    client = get_s3_client()
    bucket = get_bucket_name()

    prefix = f"models/{experiment_name}/{run_name}"

    for filename in ("model.pt", "config.yaml", "metrics.json"):
        client.upload_file(
            str(directory / filename),
            bucket,
            f"{prefix}/{filename}",
        )

    return f"s3://{bucket}/{prefix}/"


def run_experiment(config_path: Path) -> None:
    """Run training, evaluation, tracking, and publication."""
    config = read_config(config_path)

    experiment_name = str(config["experiment_name"])
    run_name = str(config["run_name"])
    seed = int(config["seed"])

    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]

    s3_path = str(data_config["s3_path"])
    max_train_samples = data_config.get("max_train_samples")
    max_val_samples = data_config.get("max_val_samples")

    model_type = str(model_config["type"])
    channels = model_config.get("params", {}).get("channels")

    epochs = int(train_config["epochs"])
    batch_size = int(train_config["batch_size"])
    learning_rate = float(train_config["lr"])

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning rate must be positive")

    random.seed(seed)
    torch.manual_seed(seed)

    tracking_uri = __import__("os").environ.get(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"Experiment: {experiment_name}")
    print(f"Run: {run_name}")
    print(f"Tracking URI: {tracking_uri}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_config(config))
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("data_version", s3_path)
        mlflow.set_tag("device", "cpu")

        print("Preparing training data...")
        train_dataset = CIFAR10S3Dataset(
            s3_path=s3_path,
            split="train",
            max_samples=max_train_samples,
        )

        print("Preparing validation data...")
        val_dataset = CIFAR10S3Dataset(
            s3_path=s3_path,
            split="val",
            max_samples=max_val_samples,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = build_model(
            model_type=model_type,
            channels=channels,
        )
        model.to("cpu")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_val: EvaluationResult | None = None

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            checkpoint_path = output_dir / "model.pt"

            training_start = time.perf_counter()

            for epoch in range(1, epochs + 1):
                train_loss = train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                )

                validation = evaluate(
                    model,
                    val_loader,
                    criterion,
                )

                mlflow.log_metric(
                    "train_loss",
                    train_loss,
                    step=epoch,
                )
                mlflow.log_metric(
                    "val_loss",
                    validation.loss,
                    step=epoch,
                )
                mlflow.log_metric(
                    "val_accuracy",
                    validation.accuracy,
                    step=epoch,
                )
                mlflow.log_metric(
                    "val_macro_f1",
                    validation.macro_f1,
                    step=epoch,
                )

                print(
                    f"Epoch {epoch}/{epochs}: "
                    f"train_loss={train_loss:.4f} "
                    f"val_loss={validation.loss:.4f} "
                    f"val_acc={validation.accuracy:.4f} "
                    f"val_f1={validation.macro_f1:.4f}"
                )

                if validation.loss < best_val_loss:
                    best_val_loss = validation.loss
                    best_epoch = epoch
                    best_val = validation

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "best_epoch": epoch,
                        },
                        checkpoint_path,
                    )

            training_seconds = time.perf_counter() - training_start

            if best_val is None:
                raise RuntimeError("No validation checkpoint created")

            mlflow.log_metric(
                "training_seconds",
                training_seconds,
            )
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metric("best_val_loss", best_val.loss)
            mlflow.log_metric(
                "best_val_accuracy",
                best_val.accuracy,
            )
            mlflow.log_metric(
                "best_val_macro_f1",
                best_val.macro_f1,
            )

            print(f"Best checkpoint: epoch {best_epoch}")

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            print("Preparing final test data...")
            test_dataset = CIFAR10S3Dataset(
                s3_path=s3_path,
                split="test",
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )

            test_result = evaluate(
                model,
                test_loader,
                criterion,
            )

            mlflow.log_metric("test_loss", test_result.loss)
            mlflow.log_metric(
                "test_accuracy",
                test_result.accuracy,
            )
            mlflow.log_metric(
                "test_macro_f1",
                test_result.macro_f1,
            )

            confusion_path = output_dir / "val_confusion_matrix.png"
            save_confusion_matrix(
                best_val.confusion_matrix,
                confusion_path,
            )

            metrics = {
                "best_epoch": best_epoch,
                "training_seconds": training_seconds,
                "val": {
                    "loss": best_val.loss,
                    "accuracy": best_val.accuracy,
                    "macro_f1": best_val.macro_f1,
                },
                "test": {
                    "loss": test_result.loss,
                    "accuracy": test_result.accuracy,
                    "macro_f1": test_result.macro_f1,
                },
            }

            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )

            shutil.copyfile(
                config_path,
                output_dir / "config.yaml",
            )

            mlflow.log_artifact(str(confusion_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(output_dir / "config.yaml"))
            mlflow.log_artifact(str(checkpoint_path))

            model_s3_path = upload_results(
                experiment_name,
                run_name,
                output_dir,
            )

            mlflow.set_tag("model_s3_path", model_s3_path)

            print(
                f"Test: loss={test_result.loss:.4f} "
                f"accuracy={test_result.accuracy:.4f} "
                f"macro_f1={test_result.macro_f1:.4f}"
            )
            print(f"Model uploaded: {model_s3_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    arguments = parser.parse_args()

    run_experiment(arguments.config)


if __name__ == "__main__":
    main()
