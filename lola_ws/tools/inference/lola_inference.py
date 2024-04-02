# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

import sys

sys.path.append("../../../tasks/eval_harness/")

from evaluate import *
from megatron import get_args
from megatron import get_tokenizer
from megatron.text_generation_utils import top_k_logits
import deepspeed

class LOLAInference(EvalHarnessAdaptor):
    def infer_batch(self, query_list, disable_tqdm=True):
        requests = []
        for context in query_list:
            if context == "":
                # end of text as context
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = self.tokenizer_encode(context)

            continuation_enc = self.tokenizer_encode('')

            requests.append(((context, ''), context_enc, continuation_enc))

        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        self.model.eval()
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen
                    if not self.adaptive_seq_len:
                        padding_length = self.max_length
                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))

                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1).cpu()  # [batch, seq, vocab]

                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                        contlen = len(cont_toks)
                        #logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        logits = logits.unsqueeze(0)  # [1, seq, vocab]
                        #greedy_tokens = logits.argmax(dim=-1)
                        # res.append(self.tokenizer.tokenizer.decode(greedy_tokens.data[0].tolist()))
                        res.append(self.tokenizer.tokenizer.decode([logits.argmax(dim=-1)[-1][-1].tolist()]))

        return ''.join(res)
    
    
    def infer_single(self, context, /, temperature=1.0, top_k=50, top_p=0.95):
        if context == "":
            # end of text as context
            context_enc = [self.EOT_TOKEN_ID]
        else:
            context_enc = self.tokenizer_encode(context)
        
        res = []
        self.model.eval()
        with torch.no_grad():
            # when too long to fit in context, truncate from the left
            inp = torch.tensor(
                (context_enc)[-(self.max_length + 1):]
                , dtype=torch.long).to(self.device)

            new_inp = inp.unsqueeze(0)

            multi_logits = self._model_call(torch.cat([new_inp], dim=0))
            
            # testing non-repetition logic
            # temperature = 1
            # top_k = 50
            # top_p = 0.95
            
            if multi_logits is not None:
                multi_logits /= temperature
                for i in range(multi_logits.shape[0]):
                    multi_logits[i] = top_k_logits(multi_logits[i], top_k, top_p)
                    multi_logits[i] = F.softmax(multi_logits[i], dim=-1)
                for logits in multi_logits :
                    # logits = logits.unsqueeze(0)  # [1, seq, vocab]
                    pred_tokens = torch.multinomial(logits, num_samples=1)
                    res.append(self.tokenizer.tokenizer.decode(pred_tokens[-1].tolist()))
                    #res.append(self.tokenizer.tokenizer.decode([logits.argmax(dim=-1)[-1][-1].tolist()]))

        return ''.join(res)
    
    def fetch_last_hidden_states(self, inps, fetch_contextual=False):
        args = get_args()
        # self.model.set_batch_fn(self.create_model_inputs)
        # round up to multiple of micro_batch_size
        new_size = ((len(inps) + args.micro_batch_size-1)  // args.micro_batch_size) * args.micro_batch_size
        padded = F.pad(inps, (0, 0, 0, new_size-len(inps)), value = 0)
        # dummy data iterator for pipelining.
        data_iterator = list((torch.stack(inp) for inp in utils.chunks(padded, args.micro_batch_size)))
        self.model.micro_batches = len(data_iterator)
        # output = self.model.eval_batch(iter(data_iterator), compute_loss = False, reduce_output = None)
        output = []
        for tokens in data_iterator:
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                                                        tokens,
                                                        self.EOT_TOKEN_ID,
                                                        args.reset_position_ids,
                                                        args.reset_attention_mask,
                                                        args.eod_mask_loss)
            if fetch_contextual:
                a_output, *other_losses = self.model(tokens,
                    position_ids,
                    attention_mask,
                    tokentype_ids=None,
                    output_last_hidden_states=True)
            else:
                a_output = self.model.module.language_model.embedding(tokens, position_ids, tokentype_ids=None)
            output.append(a_output)
        
        return output
    
    def get_token_embeddings(self, context, fetch_contextual=False):
        if context == "":
            # end of text as context
            context_enc = [self.EOT_TOKEN_ID]
        else:
            context_enc = self.tokenizer_encode(context)

        token_units = self.tokenizer.tokenizer.tokenize(context)

        token_embeddings = []

        len_context = len(context_enc)
        
        self.model.eval()
        with torch.no_grad():
            # when too long to fit in context, truncate from the left
            inp = torch.tensor(
                (context_enc)[-(self.max_length + 1):]
                , dtype=torch.long).to(self.device)

            new_inp = inp.unsqueeze(0)

            embeddings_tensor = self.fetch_last_hidden_states(torch.cat([new_inp], dim=0), fetch_contextual)
            embedding_list = embeddings_tensor[0].tolist()

            for i in range(len_context):
                token_embeddings.append({
                    'token' : token_units[i],
                    'tokenid': context_enc[i],
                    'embedding': embedding_list[i][0]
                })

            return token_embeddings


def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Model inference options')
    # default arg for evaluate.py
    group.add_argument('--adaptive_seq_len',  default = True, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--num_fewshot', type=int, default = 0, help='Number of few-shot prompts.')
    group.add_argument('--eval_fp32',  default = True, action='store_true', help='Should the evaluation run in fp32')
    return parser

def generate_output(input_text, max_sentences, infer_tool: LOLAInference, max_tokens=500, **kwargs):
    generated_text = ''
    sent_count = 0
    for i in range(max_tokens):
        # generated_text = infer_tool.infer_batch([input_text + ' '])
        generated_text = infer_tool.infer_single(input_text, **kwargs)
        input_text+=generated_text
        if generated_text == '.':
            sent_count+=1
            if sent_count >= max_sentences:
                break
    return input_text

def main():

    model = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)

    args = get_args()
    
    if args.deepspeed and args.adaptive_seq_len:
        # adaptive_seq_len hack #1:
        # CL automatically enables reset_activation_shape() which allows us to change input shapes
        # and it also reshapes the attenion scores in attention_mask_func
        args.curriculum_learning_legacy = 1
    
    model.eval()
    model.module.activation_checkpoint_interval = 0
    model._compute_loss = False
    model.fwd_outputs = []

    # import megatron.model as mm
    # engine = deepspeed.init_inference(model=model,
    #                                   mp_size=1,
    #                                   tensor_parallel={"mpu": mpu},
    #                                   dtype=torch.half,
    #                                   replace_with_kernel_inject=True,
    #                                   moe_experts=[16],
    #                                   moe_type='standard')
    
    # model = engine.module
    
    # print(model)
    # output = model('Input String')
    # print(output)

    tokenizer = get_tokenizer()

    infer_tool = LOLAInference(model, tokenizer)
    
    # input_text = "O mar enrola na areia"
    input_text = "The quick brown fox"
    #input_text = "Hallo! Ich bin Sven, I komme aus"
    #input_text = "Привет, меня зовут Иван"
    # input_text = "Question: To make Belgian waffles\nAnswer:"
    
    output_text = generate_output(input_text, 2, infer_tool, temperature=1.0, top_k=50, top_p=0.95)

    embeddings_1 = infer_tool.get_token_embeddings(input_text)

    embeddings_2 = infer_tool.get_token_embeddings(input_text, True)
    
    print("Generated text: ", output_text)
    

if __name__ == '__main__':
    main()
