import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import (ensure_directories, extract_behavioral_property,
                   load_config, set_seed)


def evaluate(
    test_original_responses: list,
    test_hypothetical_responses: list,
    model_name: str = "gpt-4o",
    n_shots: int = 10,
) -> pd.DataFrame:
    """Evaluate hypothetical response accuracy using test data responses"""
    results = []
    for original_response, hypothetical_response in zip(
        test_original_responses, test_hypothetical_responses
    ):
        correct_answer = extract_behavioral_property(
            original_response["response"],
            original_response["behavioral_property"],
            original_response["option_matching_ethical_stance"],
        )
        is_correct = hypothetical_response["response"].lower().strip() == correct_answer

        results.append({"model": model_name, "n_shots": n_shots, "correct": is_correct})

    # Create DataFrame
    df = pd.DataFrame(results)
    accuracy_df = df.groupby(["model", "n_shots"])["correct"].mean().reset_index()

    return accuracy_df


def plot_accuracy(accuracy_results: pd.DataFrame, output_path: str) -> None:
    """Plot accuracy curve for different n_shots"""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=accuracy_results, x="n_shots", y="correct", marker="o")
    plt.title("Hypothetical Response Accuracy by Number of Shots")
    plt.xlabel("Number of Shots")
    plt.ylabel("Accuracy")

    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    set_seed()
    config = load_config()
    ensure_directories(config)
    model_name = config["model"]["name"]

    test_original_responses_path = (
        Path(config["paths"]["responses_dir"]) / model_name / "test" / "original.json"
    )
    with open(test_original_responses_path, "r") as f:
        test_original_responses = json.load(f)

    all_results = []
    for n_shots in config["few_shot"]["n_shots_list"]:
        test_hypothetical_responses_path = (
            Path(config["paths"]["responses_dir"])
            / model_name
            / "test"
            / f"hypothetical_{n_shots}shot.json"
        )
        with open(test_hypothetical_responses_path, "r") as f:
            test_hypothetical_responses = json.load(f)

        results = evaluate(
            test_original_responses,
            test_hypothetical_responses,
            model_name,
            n_shots=n_shots,
        )
        all_results.append(results)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Plot accuracy curve
    plot_accuracy(
        combined_results,
        Path(config["paths"]["figures_dir"]) / model_name / "accuracy.png",
    )
