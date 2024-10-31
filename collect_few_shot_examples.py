import json
from pathlib import Path

from tqdm import tqdm

from utils import (ensure_directories, extract_behavioral_property,
                   get_model_response, load_config, load_dataset, set_seed)


def collect_few_shot_examples(train_data: list, model_name: str) -> None:
    """Collect and save original responses for train data with few-shot learning"""
    few_shot_prompt = []
    few_shot_examples_path = (
        Path(config["paths"]["processed_dir"]) / model_name / "few_shot_examples.json"
    )

    for row in tqdm(train_data, desc="Collecting train responses"):
        # Get original response using current few_shot_prompt
        messages = few_shot_prompt + [m.dict() for m in row.object_level_prompt]
        response = get_model_response(messages=messages, model_name=model_name)

        # Extract behavioral property and update few_shot_prompt
        hypothetical_answer = extract_behavioral_property(
            response,
            row.behavioral_property,
            row.option_matching_ethical_stance,
        )
        few_shot_prompt.extend(
            [
                {"role": "user", "content": row.hypothetical_prompt[0].content},
                {"role": "assistant", "content": hypothetical_answer},
            ]
        )

        # Save responses after each update
        with open(few_shot_examples_path, "w") as f:
            json.dump(few_shot_prompt, f, indent=2)


if __name__ == "__main__":
    set_seed()
    config = load_config()
    ensure_directories(config)
    train_data = load_dataset(
        config["paths"]["train_dir"], config["n_samples"]["train"]
    )
    test_data = load_dataset(config["paths"]["test_dir"], config["n_samples"]["test"])
    collect_few_shot_examples(train_data, config["model"]["name"])
