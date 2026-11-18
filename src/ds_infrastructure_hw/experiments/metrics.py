"""Evaluation metrics for image classification."""

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EvaluationResult:
    """Metrics and confusion matrix for one dataset split."""

    loss: float
    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    num_classes: int = 10,
) -> EvaluationResult:
    """Evaluate a model without updating its weights."""
    model.eval()

    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = cast(torch.Tensor, model(images))
            loss = cast(torch.Tensor, criterion(logits, labels))

            predictions = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for true_label, prediction in zip(
                labels.tolist(),
                predictions.tolist(),
                strict=True,
            ):
                confusion[int(true_label)][int(prediction)] += 1

    if total_samples == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    correct = sum(confusion[i][i] for i in range(num_classes))

    f1_scores: list[float] = []

    for class_id in range(num_classes):
        true_positive = confusion[class_id][class_id]

        false_positive = sum(
            confusion[row][class_id] for row in range(num_classes) if row != class_id
        )

        false_negative = sum(
            confusion[class_id][column] for column in range(num_classes) if column != class_id
        )

        denominator = 2 * true_positive + false_positive + false_negative

        f1 = 2 * true_positive / denominator if denominator > 0 else 0.0

        f1_scores.append(f1)

    return EvaluationResult(
        loss=total_loss / total_samples,
        accuracy=correct / total_samples,
        macro_f1=sum(f1_scores) / num_classes,
        confusion_matrix=confusion,
    )
