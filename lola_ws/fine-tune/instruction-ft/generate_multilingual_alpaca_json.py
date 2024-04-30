import json
import requests
from tqdm import tqdm
from random import shuffle

# List of URLs containing JSON arrays
urls = [
    "https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.bg.json",
    "https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.cs.json",
    "https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.de.json",
    "https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.en.json",
    "https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.es.json",
    "https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.fi.json",
    "https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.fr.json",
    "https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.pt.json",
    "https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.ru.json",
    "https://raw.githubusercontent.com/hplt-project/monolingual-multilingual-instruction-tuning/main/training-data/alpaca_data_cleaned.zh.json"
]

# Filename to write the combined JSON
filename = "alpaca_multilingual.json"

def download_json(url):
    """Download and return JSON data from the given URL."""
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    return response.json()

def main():
    all_data = []
    
    # Download data from each URL and extend the main list
    for url in tqdm(urls, desc="Downloading JSON data"):
        try:
            data = download_json(url)
            if isinstance(data, list):  # Ensure it is a list to combine
                all_data.extend(data)
            else:
                print(f"Warning: Data from {url} is not a list and will be skipped.")
        except requests.HTTPError as e:
            print(f"Failed to download data from {url}: {str(e)}")

    # Shuffle the combined data
    shuffle(all_data)

    # Write the combined and shuffled data to a file
    with open(filename, "w") as f:
        json.dump(all_data, f, indent=4)

    print(f"Data combined, shuffled, and written to '{filename}'.")

if __name__ == "__main__":
    main()
