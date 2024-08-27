import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import lang2vec.lang2vec as l2v
# output dir
output_dir = './ling_distance'
# Read the language list
langlist_file_path = '../languages.txt'
langmap_file_path = 'culturax-iso639_3-map.json'

distances_to_compute = ['syntactic','geographic','phonological','genetic','inventory','featural']

def linguistic_distance(lang1, lang2):
    # Example: returns a tuple of two distances
    return l2v.distance(distances_to_compute, lang1, lang2)

with open(langlist_file_path, 'r') as file, open(langmap_file_path, 'r') as langmap_file:
    language_codes = file.read().splitlines()
    lang_map = json.load(langmap_file)

size = len(language_codes)
num_distances = len(distances_to_compute)
dist_matrices = [np.zeros((size, size)) for _ in range(num_distances)]

pbar = tqdm(total=size*size)

# for each language in the languages.txt
for i in range(size):
    culturax_main_lang = language_codes[i]
    iso639_main_lang = lang_map[culturax_main_lang]
    # for each language again
    for j in range(i, size): # Only compute for the upper triangular part (including diagonal)
        culturax_compare_lang = language_codes[j]
        iso639_comp_lang = lang_map[culturax_compare_lang]
        if iso639_main_lang and iso639_comp_lang:
            # find distance between these languages
            distances = linguistic_distance(iso639_main_lang, iso639_comp_lang)
            for k in range(num_distances):
                dist_matrices[k][i][j] = distances[k]
                dist_matrices[k][j][i] = distances[k]  # Copy the value to the lower triangular part
        pbar.update(2)
pbar.close()

# create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)
# export the distances
for k in range(num_distances):
    dist_df = pd.DataFrame(dist_matrices[k], index=language_codes, columns=language_codes)
    dist_df.to_csv(f'linguistic_distance_matrix_{distances_to_compute[k]}.csv')