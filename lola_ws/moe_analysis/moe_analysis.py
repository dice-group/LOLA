#!/usr/bin/env python3
import logging
logging.basicConfig(level=logging.INFO)

import argparse
import csv
import datasets
import itertools
import json
import pathlib
import torch
import tqdm
import transformers

import sys
sys.path.append("../")
from lola_hf_model.modeling_lola_gpt2 import LOLALMHeadModel
import lola_hf_model.modeling_lola_gpt2

def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch
itertools.batched = batched

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='(example: dice-research/lola_v1')
parser.add_argument('--lang', required=True, help='(example: en)')
parser.add_argument('--rows-limit', type=int, required=True)
parser.add_argument('--output-dir', type=pathlib.Path, required=True)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--batch-size', type=int, default=1)
args = parser.parse_args()

logging.info('Args: {}', args)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
model = LOLALMHeadModel.from_pretrained(args.model).to(args.device)

dataset = datasets.load_dataset('uonlp/CulturaX', args.lang, split='train', streaming=True)

def tokenize(text):
    'Returns a tuple: fixed-length tensor with token IDs and the actual (possibly truncated) length in tokens'
    ids = tokenizer(
        text,
        truncation=True,
        return_tensors='pt',
    )['input_ids']
    return torch.nn.functional.pad(ids, (0, tokenizer.model_max_length - ids.shape[1]), value=tokenizer.pad_token_id), min(ids.shape[1], tokenizer.model_max_length)

args.output_dir.mkdir(parents=True, exist_ok=True)
with open(args.output_dir / f'row_data.{args.lang}.jsonl', 'w') as datafile, open(args.output_dir / f'token_experts.{args.lang}.dat', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for batch in itertools.batched(enumerate(tqdm.tqdm(itertools.islice(dataset, args.rows_limit), desc=args.lang, total=args.rows_limit)), args.batch_size):
        batch = list(batch)
        experts_info = []
        lola_hf_model.modeling_lola_gpt2.expert_analysis_callback = lambda selected_experts: experts_info.append([e[0] for e in selected_experts])
        input_ids, input_lenghts = zip(*(tokenize(row['text']) for _, row in batch))
        input_ids = torch.cat(tuple(input_ids)).to(args.device)
        _ = model.generate(
            input_ids=input_ids,
            max_new_tokens=1, # transformers does not accept 0
        )
        layers = len(experts_info)
        assert layers == 12
        for i, (index, row) in enumerate(batch):
            if index == 0:
                writer.writerow(['lang', 'row', 'token'] + [f'expert_layer_{layer}' for layer in range(layers)])
            datafile.write(json.dumps(row | {'index': index}) + '\n')
            for token in range(input_lenghts[i]):
                writer.writerow([args.lang, index, token] + [experts_info[layer][i * tokenizer.model_max_length + token].item() for layer in range(layers)])
