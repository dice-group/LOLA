import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import lang2vec.lang2vec as l2v
from multiprocessing import Pool, cpu_count

# Output dir
output_dir = './ling_distance'
# Read the language list
langlist_file_path = '../languages.txt'
langmap_file_path = 'culturax-iso639_3-map.json'

distances_to_compute = ['syntactic','geographic','phonological','genetic','inventory','featural']

def linguistic_distance(lang1, lang2):
    # Example: returns a tuple of distances
    return l2v.distance(distances_to_compute, lang1, lang2)

def compute_distances(args):
    i, j, iso639_main_lang, iso639_comp_lang, num_distances = args
    if iso639_main_lang and iso639_comp_lang:
        distances = linguistic_distance(iso639_main_lang, iso639_comp_lang)
        return i, j, distances
    else:
        return i, j, [0.0] * num_distances

if __name__ == "__main__":
    with open(langlist_file_path, 'r') as file, open(langmap_file_path, 'r') as langmap_file:
        language_codes = file.read().splitlines()
        lang_map = json.load(langmap_file)

    size = len(language_codes)
    num_distances = len(distances_to_compute)
    dist_matrices = [np.zeros((size, size)) for _ in range(num_distances)]

    pbar = tqdm(total=size * (size + 1) // 2)

    # Prepare tasks
    tasks = []
    for i in range(size):
        culturax_main_lang = language_codes[i]
        iso639_main_lang = lang_map[culturax_main_lang]
        for j in range(i, size):
            culturax_compare_lang = language_codes[j]
            iso639_comp_lang = lang_map[culturax_compare_lang]
            tasks.append((i, j, iso639_main_lang, iso639_comp_lang, num_distances))

    # Utilize multiprocessing Pool to parallelize the task
    with Pool(cpu_count()) as pool:
        for i, j, distances in pool.imap_unordered(compute_distances, tasks):
            for k in range(num_distances):
                dist_matrices[k][i][j] = distances[k]
                dist_matrices[k][j][i] = distances[k]  # Copy the value to the lower triangular part
            pbar.update(1)
    pbar.close()

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    # Export the distances
    for k in range(num_distances):
        dist_df = pd.DataFrame(dist_matrices[k], index=language_codes, columns=language_codes)
        dist_df.to_csv(f'{output_dir}/linguistic_distance_matrix_{distances_to_compute[k]}.csv')
