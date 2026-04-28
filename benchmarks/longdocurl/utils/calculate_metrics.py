import argparse
import json
from pathlib import Path

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).absolute().parent.parent))

from utils.utils_score_v3 import eval_score

def calculate_accuracy(answers: list, annotations: list, answer_formats: list):
    total_scores = 0.0
    for pred_ans, annotation, answer_format in zip(answers, annotations, answer_formats):
        if pred_ans == "Fail to extract":
            score_v3 = 0.0
        else:
            score_v3 = eval_score(annotation, pred_ans, answer_format)
        
        total_scores += score_v3
    
    generalized_score = total_scores / len(answers)

    return generalized_score


def calculate_metrics(results_file: str):
    results_file = Path(results_file)
    with open(results_file, "r", encoding="utf-8") as rf:
        samples = [json.loads(_.strip()) for _ in rf.readlines()]

    for sample in samples:
        assert "pred" in sample

    answers = [_["pred"] for _ in samples]
    annotations = [_["answer"] for _ in samples]
    answer_formats = [_["answer_format"] for _ in samples]

    generalized_score = calculate_accuracy(answers, annotations, answer_formats) # calculate on size of successful samples
    rectified_generalized_score = generalized_score * len(answers) / 2325 # calculate on size of 2325

    output_dir = results_file.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "metrics.json"

    metrics = {
        "avg_acc": generalized_score,
        "rectified_avg_acc": rectified_generalized_score,
    }

    with open(output_file, "w", encoding="utf-8") as wf:
        json.dump(metrics, wf, ensure_ascii=False, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, default="")

    args = parser.parse_args()
    metrics = calculate_metrics(args.results_file)

    print("--------------------------------------")
    print("Avg. Acc: {}".format(metrics["avg_acc"]))
    print("Rectified Avg. Acc: {}".format(metrics["rectified_avg_acc"]))
