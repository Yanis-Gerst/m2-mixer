import torch
from torchmetrics import Accuracy, F1Score, Precision, Recall
from typing import List, Dict

device = "cuda" if torch.cuda.is_available() else "cpu"

# preds_names = ["pure", "low", "medium", "hard"]
preds_names = ["pure_framework"]
preds_path = [f"preds/pred_{name}.pt" for name in preds_names]
for path in preds_path:

    loaded_data = torch.load(
        path, map_location=device)
    print(loaded_data)
    def setup_metrics_for_evaluation() -> Dict[str, torch.nn.Module]:

        metrics = dict(
            acc=Accuracy(task="multiclass", num_classes=10,).to(device),
            f1m=F1Score(task="multiclass", num_classes=10,
                        average='macro').to(device),
            prec_m=Precision(task="multiclass", num_classes=10,
                             average='macro').to(device),
            rec_m=Recall(task="multiclass", num_classes=10,
                         average='macro').to(device)
        )

        return metrics

    evaluation_metrics = setup_metrics_for_evaluation()

    predictions = loaded_data['preds']
    true_labels = loaded_data['labels']

    calculated_scores = {}
    print(f"Calculating scores... for {path}")
    for metric_name, metric_calculator in evaluation_metrics.items():
        metric_calculator.update(predictions, true_labels)
        score = metric_calculator.compute()
        calculated_scores[metric_name] = score
        print(f"{metric_name}: {score.item():.4f}")

    metrics = {}
    other_metrics = ["aleatoric_uncertainty", "epidemic_uncertainty"]
    if other_metrics[0] in loaded_data:
        for metric_name in other_metrics:
            print(f"{metric_name}: {loaded_data[metric_name].mean():.4f}")
