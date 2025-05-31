# Load the AYA Dataset from Huggingface (CohereLabs/aya_dataset)
# Load the "default" subset and "train" split 
# Extract the fields:  inputs, targets, language
# Map the aya fields to alpaca fields: inputs -> instruction, targets -> output
# Write it to alpaca json array with the following fields:
# {
#     "instruction": "Give three tips for staying healthy.",
#     "input": "",
#     "output": "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
# }

import json
from datasets import load_dataset
from tqdm import tqdm

def convert_aya_to_alpaca():
    # Load the AYA Dataset from Huggingface (CohereLabs/aya_dataset)
    dataset = load_dataset('CohereLabs/aya_dataset', 'default')

    # Extract the "train" split
    train_split = dataset['train']

    # Extract fields:  inputs, targets, language
    data_points = []

    for entry in tqdm(train_split, desc="Converting to Alpaca format"):
        input_text = entry['inputs']
        output_text = entry['targets']
        language = entry['language']

        # Map the aya fields to alpaca fields
        alpaca_entry = {
            "instruction": input_text.strip(),
            "input": "",
            "output": output_text.strip(),
            "language": language
        }

        data_points.append(alpaca_entry)

    # Write it to alpaca json array
    with open('../aya_alpaca_dataset.json', 'w') as f:
        json.dump(data_points, f, indent=2)

if __name__ == "__main__":
    convert_aya_to_alpaca()