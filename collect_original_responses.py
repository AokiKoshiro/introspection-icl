import json
from pathlib import Path

from tqdm import tqdm

from utils import (ensure_directories, get_model_response, load_config,
                   load_dataset, set_seed)


def collect_original_responses(
    train_data: list, test_data: list, model_name: str
) -> None:
    """Collect and save original responses for train and test data"""
    # Process train data
    train_responses = []
    for row in tqdm(train_data, desc="Collecting train responses"):
        response = get_model_response(
            messages=[m.dict() for m in row.object_level_prompt], model_name=model_name
        )
        train_responses.append(
            {
                "original_question": row.original_question,
                "response": response,
                "behavioral_property": row.behavioral_property,
                "option_matching_ethical_stance": row.option_matching_ethical_stance,
            }
        )

    # Process test data
    test_responses = []
    for row in tqdm(test_data, desc="Collecting test responses"):
        response = get_model_response(
            messages=[m.dict() for m in row.object_level_prompt], model_name=model_name
        )
        test_responses.append(
            {
                "original_question": row.original_question,
                "response": response,
                "behavioral_property": row.behavioral_property,
                "option_matching_ethical_stance": row.option_matching_ethical_stance,
            }
        )

    # Save responses
    train_output_path = (
        Path(config["paths"]["responses_dir"]) / model_name / "train" / "original.json"
    )
    with open(train_output_path, "w") as f:
        json.dump(train_responses, f, indent=2)

    test_output_path = (
        Path(config["paths"]["responses_dir"]) / model_name / "test" / "original.json"
    )
    with open(test_output_path, "w") as f:
        json.dump(test_responses, f, indent=2)


if __name__ == "__main__":
    set_seed()
    config = load_config()
    ensure_directories(config)
    train_data = load_dataset(
        config["paths"]["train_dir"], config["n_samples"]["train"]
    )
    test_data = load_dataset(config["paths"]["test_dir"], config["n_samples"]["test"])
    collect_original_responses(
        train_data,
        test_data,
        config["model"]["name"],
    )
