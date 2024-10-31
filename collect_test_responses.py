import json
from pathlib import Path

from tqdm import tqdm

from utils import (ensure_directories, get_model_response, load_config,
                   load_dataset, set_seed)


def collect_test_responses(
    test_data: list,
    few_shot_examples: list,
    model_name: str,
    n_shots: int = 10,
) -> None:
    """Collect and save both original and hypothetical responses for test data"""
    test_responses = []

    for i, row in enumerate(
        tqdm(test_data, desc=f"Collecting test responses ({n_shots} shots)")
    ):
        # Initialize few_shot_prompt with first n_shots examples
        few_shot_prompt = few_shot_examples[: n_shots * 2]

        # Get original response
        original_prompt = few_shot_prompt + [m.dict() for m in row.object_level_prompt]
        original_response = get_model_response(
            messages=original_prompt, model_name=model_name
        )

        # Get hypothetical response
        hypothetical_prompt = few_shot_prompt + [
            {"role": "user", "content": row.hypothetical_prompt[0].content}
        ]
        hypothetical_response = get_model_response(
            messages=hypothetical_prompt, model_name=model_name
        )

        test_responses.append(
            {
                "original_question": row.original_question,
                "behavioral_property": row.behavioral_property,
                "original_response": original_response,
                "hypothetical_response": hypothetical_response,
            }
        )

    # Save responses
    test_responses_path = (
        Path(config["paths"]["processed_dir"])
        / model_name
        / "test_responses"
        / f"{n_shots}shot.json"
    )

    with open(test_responses_path, "w") as f:
        json.dump(test_responses, f, indent=2)


if __name__ == "__main__":
    set_seed()
    config = load_config()
    ensure_directories(config)

    model_name = config["model"]["name"]
    train_data = load_dataset(
        config["paths"]["train_dir"], config["n_samples"]["train"]
    )
    test_data = load_dataset(config["paths"]["test_dir"], config["n_samples"]["test"])

    few_shot_examples_path = (
        Path(config["paths"]["processed_dir"]) / model_name / "few_shot_examples.json"
    )
    with open(few_shot_examples_path, "r") as f:
        few_shot_examples = json.load(f)

    for n_shots in config["few_shot"]["n_shots_list"]:
        collect_test_responses(
            test_data,
            few_shot_examples,
            model_name,
            n_shots,
        )
