import json
from pathlib import Path

from tqdm import tqdm

from utils import (ensure_directories, get_model_response, load_config,
                   load_dataset, set_seed)


def collect_original_responses(data: list, model_name: str, data_type: str) -> None:
    """Collect and save original responses for the given dataset

    Args:
        data: List of data samples to process
        model_name: Name of the model to use
        data_type: Type of data ('train' or 'test')
    """
    responses = []
    output_path = (
        Path(config["paths"]["responses_dir"])
        / model_name
        / data_type
        / "original.json"
    )

    for row in tqdm(data, desc=f"Collecting {data_type} responses"):
        response = get_model_response(
            messages=[m.dict() for m in row.object_level_prompt], model_name=model_name
        )
        responses.append(
            {
                "original_question": row.original_question,
                "response": response,
                "behavioral_property": row.behavioral_property,
                "option_matching_ethical_stance": row.option_matching_ethical_stance,
            }
        )
        with open(output_path, "w") as f:
            json.dump(responses, f, indent=2)


if __name__ == "__main__":
    set_seed()
    config = load_config()
    ensure_directories(config)
    train_data = load_dataset(
        config["paths"]["train_dir"], config["n_samples"]["train"]
    )
    test_data = load_dataset(config["paths"]["test_dir"], config["n_samples"]["test"])
    collect_original_responses(train_data, config["model"]["name"], "train")
    collect_original_responses(test_data, config["model"]["name"], "test")
