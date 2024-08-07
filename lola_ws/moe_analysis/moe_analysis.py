#!/usr/bin/env python3
import logging
logging.basicConfig(level=logging.INFO)

import argparse
import csv
import datasets
import itertools
import json
import pathlib
import tqdm
import transformers

import sys
sys.path.append("../")
from lola_hf_model.modeling_lola_gpt2 import LOLALMHeadModel
import lola_hf_model.modeling_lola_gpt2

MAX_SEQ_LENGTH = 2048

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', required=True, help='(example: en)')
parser.add_argument('--rows-limit', type=int, required=True)
parser.add_argument('--output-dir', type=pathlib.Path, required=True)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()

tokenizer = transformers.AutoTokenizer.from_pretrained('dice-research/lola_v1')
model = LOLALMHeadModel.from_pretrained('dice-research/lola_v1').to(args.device)

dataset = datasets.load_dataset('uonlp/CulturaX', args.dataset_name, split='train', streaming=True)

args.output_dir.mkdir(parents=True, exist_ok=True)
with open(args.output_dir / f'row_data.{args.dataset_name}.data', 'w') as datafile, open(args.output_dir / f'token_experts.{args.dataset_name}.dat', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for i, row in enumerate(tqdm.tqdm(itertools.islice(dataset, args.rows_limit), desc=args.dataset_name, total=args.rows_limit)):
        experts_info = []
        lola_hf_model.modeling_lola_gpt2.expert_analysis_callback = lambda selected_experts: experts_info.append([e[0] for e in selected_experts])
        input_ids = tokenizer(row['text'], return_tensors='pt')['input_ids'][:, :MAX_SEQ_LENGTH].to(model.device)
        _ = model.generate(
            input_ids=input_ids,
            max_new_tokens=1, # transformers does not accept 0
        )
        tokens = len(input_ids[0])
        layers = len(experts_info)
        if i == 0:
            writer.writerow(['dataset', 'row', 'token'] + [f'expert_layer_{layer}' for layer in range(layers)])
        datafile.write(json.dumps(row | {'index': i}) + '\n')
        for token in range(tokens):
            writer.writerow([args.dataset_name, i, token] + [int(experts_info[layer][token]) for layer in range(layers)])
