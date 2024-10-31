import json
import random
from pathlib import Path

from tqdm import tqdm

from utils import (ensure_directories, extract_behavioral_property,
                   get_model_response, load_config, load_dataset, set_seed)


def create_examples(train_data: list, train_original_responses: list) -> list:
    examples = []
    for train_item, response_item in zip(train_data, train_original_responses):
        correct_answer = extract_behavioral_property(
            response_item["response"],
            response_item["behavioral_property"],
            response_item["option_matching_ethical_stance"],
        )
        examples.append(
            {
                "hypothetical_prompt": train_item.hypothetical_prompt[0].content,
                "correct_answer": correct_answer,
            }
        )

    return examples


def create_few_shot_prompt(
    examples: list, target_question: list, n_shots: int = 10, seed: int = 42
) -> list:
    """Create few-shot prompt with examples"""
    # Create a separate random number generator for this prompt
    rng = random.Random(seed)

    # Randomly select n_shots examples
    selected_examples = rng.sample(examples, n_shots)

    messages = []
    for ex in selected_examples:
        messages.extend(
            [
                {"role": "user", "content": ex["hypothetical_prompt"]},
                {"role": "assistant", "content": ex["correct_answer"]},
            ]
        )
    messages.extend([{"role": "user", "content": target_question}])
    return messages


def collect_hypothetical_responses(
    test_data: list,
    few_shot_examples: list,
    model_name: str,
    n_shots: int = 10,
) -> None:
    """Collect and save hypothetical responses with few-shot learning"""
    responses = []
    for i, row in enumerate(
        tqdm(test_data, desc=f"Collecting hypothetical responses ({n_shots} shots)")
    ):
        # Use index as seed for reproducibility
        few_shot_prompt = create_few_shot_prompt(
            few_shot_examples, row.hypothetical_prompt[0].content, n_shots, seed=i
        )

        response = get_model_response(messages=few_shot_prompt, model_name=model_name)

        responses.append(
            {
                "original_question": row.original_question,
                "response": response,
                "behavioral_property": row.behavioral_property,
            }
        )

    output_path = (
        Path(config["paths"]["responses_dir"])
        / model_name
        / "test"
        / f"hypothetical_{n_shots}shot.json"
    )
    with open(output_path, "w") as f:
        json.dump(responses, f, indent=2)


if __name__ == "__main__":
    set_seed()
    config = load_config()
    ensure_directories(config)

    model_name = config["model"]["name"]
    train_data = load_dataset(
        config["paths"]["train_dir"], config["n_samples"]["train"]
    )
    test_data = load_dataset(config["paths"]["test_dir"], config["n_samples"]["test"])

    train_original_responses_path = (
        Path(config["paths"]["responses_dir"]) / model_name / "train" / "original.json"
    )
    with open(train_original_responses_path, "r") as f:
        train_original_responses = json.load(f)

    few_shot_examples = create_examples(train_data, train_original_responses)

    for n_shots in config["few_shot"]["n_shots_list"]:
        collect_hypothetical_responses(
            test_data,
            few_shot_examples,
            model_name,
            n_shots,
        )
