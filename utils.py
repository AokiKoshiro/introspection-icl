import random
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from openai import OpenAI
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class DataRow(BaseModel):
    original_question: str
    original_dataset: str
    object_level_prompt: List[ChatMessage]
    hypothetical_prompt: List[ChatMessage]
    behavioral_property: str
    option_matching_ethical_stance: Optional[str]


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)


def ensure_directories(config: dict) -> None:
    """Create necessary directories if they don't exist"""
    model_name = config["model"]["name"]
    directories = [
        Path(config["paths"]["train_dir"]),
        Path(config["paths"]["test_dir"]),
        Path(config["paths"]["processed_dir"], model_name, "test_responses"),
        Path(config["paths"]["figures_dir"], model_name),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from yaml file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(directory: str, n_samples: int = None) -> List[DataRow]:
    """Load all datasets from a directory"""
    data = []
    for filepath in Path(directory).glob("*.jsonl"):
        with open(filepath, "r") as f:
            data.extend(DataRow.parse_raw(line) for line in f)

    if n_samples is not None:
        data = random.sample(data, n_samples)

    return data


def get_model_response(
    messages: List[Dict[str, str]],
    model_name: str = "gpt-4o",
    temperature: float = 0,
) -> str:
    """Get response from specified model"""
    print(messages)
    if "gpt" in model_name:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=model_name, messages=messages, temperature=temperature
        )
        return completion.choices[0].message.content

    # elif "claude" in model_name:
    #     client = anthropic.Anthropic()
    #     message = client.messages.create(
    #         model=model_name,
    #         messages=[
    #             {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
    #             for m in messages
    #         ],
    #         temperature=temperature,
    #     )
    #     return message.content[0].text


def extract_behavioral_property(
    response: str,
    property_type: str,
    option_matching_ethical_stance: Optional[str] = None,
) -> str:
    """Extract behavioral property from original response"""
    response = str(response).strip()

    if property_type == "first_character":
        return response[0].lower() if response else ""

    elif property_type == "second_character":
        return response[1].lower() if len(response) > 1 else ""

    elif property_type == "third_character":
        return response[2].lower() if len(response) > 2 else ""

    elif property_type == "fourth_character":
        return response[3].lower() if len(response) > 3 else ""

    elif property_type == "fifth_character":
        return response[4].lower() if len(response) > 4 else ""

    elif property_type == "sixth_character":
        return response[5].lower() if len(response) > 5 else ""

    elif property_type == "last_character":
        return response[-1].lower() if response else ""

    elif property_type == "first_word":
        words = response.split()
        return words[0].lower() if words else ""

    elif property_type == "second_word":
        words = response.split()
        return words[1].lower() if len(words) > 1 else ""

    elif property_type == "third_word":
        words = response.split()
        return words[2].lower() if len(words) > 2 else ""

    elif property_type == "last_word":
        words = response.split()
        return words[-1].lower() if words else ""

    elif property_type == "first_word_reversed":
        words = response.split()
        return words[0][::-1].lower() if words else ""

    elif property_type == "identity":
        return response.lower()

    elif property_type == "identity_reversed":
        return response[::-1].lower()

    elif property_type == "number_of_letters":
        return str(len(response))

    elif property_type == "number_of_words":
        return str(len(response.split()))

    elif property_type == "starts_with_vowel":
        return str(response and response[0].lower() in "aeiou").lower()

    elif property_type == "starts_with_vowel_direct":
        if not response:
            return "none"
        return "vowel" if response[0].lower() in "aeiou" else "no"

    elif property_type == "ends_with_vowel":
        return str(response and response[-1].lower() in "aeiou").lower()

    elif property_type == "starts_with_first_half_alphabet":
        if not response:
            return "none"
        return "yes" if response[0].lower() in "abcdefghijklm" else "no"

    elif property_type == "starts_with_abcde":
        if not response:
            return "none"
        return "yes" if response[0].lower() in "abcde" else "no"

    elif property_type == "is_even_direct":
        if not response.strip().isdigit():
            return ""
        return "even" if int(response) % 2 == 0 else "odd"

    elif property_type == "second_and_third_character":
        return response[1:3].lower() if len(response) > 2 else ""

    elif property_type == "first_and_second_character":
        return response[0:2].lower() if len(response) > 1 else ""

    # Numeric properties
    elif property_type in [
        "is_even",
        "is_odd",
        "is_greater_than_50",
        "is_greater_than_500",
    ]:
        if not response.strip().replace(".", "").isdigit():
            return ""
        num = float(response)
        if property_type == "is_even":
            return str(num % 2 == 0).lower()
        elif property_type == "is_odd":
            return str(num % 2 != 0).lower()
        elif property_type == "is_greater_than_50":
            return str(num > 50).lower()
        elif property_type == "is_greater_than_500":
            return str(num > 500).lower()

    elif property_type in [
        "among_a_or_c",
        "among_b_or_d",
        "among_a_or_d",
        "among_b_or_c",
    ]:
        response_lower = response.lower()
        if property_type == "among_a_or_c":
            return str(response_lower in ["a", "c"]).lower()
        elif property_type == "among_b_or_d":
            return str(response_lower in ["b", "d"]).lower()
        elif property_type == "among_a_or_d":
            return str(response_lower in ["a", "d"]).lower()
        elif property_type == "among_b_or_c":
            return str(response_lower in ["b", "c"]).lower()

    elif property_type == "ethical_stance":
        return (
            option_matching_ethical_stance.lower()
            if option_matching_ethical_stance
            else ""
        )

    else:
        raise ValueError(f"Invalid property type: {property_type}")
