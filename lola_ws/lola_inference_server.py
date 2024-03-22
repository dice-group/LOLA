# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

from tasks.eval_harness.evaluate import *
from megatron import get_args
from megatron import get_tokenizer
from megatron.text_generation_utils import top_k_logits

import logging
import os
# importing the flask Module
from flask import request
from flask import Flask

from threading import Lock

MUTEX = Lock()

class LOLAInference(EvalHarnessAdaptor):
    
    def reset_stuff(self):
        self.model.eval()
        self.model.module.activation_checkpoint_interval = 0
        self.model._compute_loss = False
        self.model.fwd_outputs = []

    
    def fetch_last_hidden_states(self, inps, fetch_contextual=False):
        self.reset_stuff()
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
    
    
    def infer_single(self, context, /, greedy=False, temperature=1.0, top_k=50, top_p=0.95):
        self.reset_stuff()
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
            
            if multi_logits is not None:
                if greedy:
                    multi_logits = F.log_softmax(multi_logits, dim=-1)
                    for logits in  multi_logits :
                        res.append(self.tokenizer.tokenizer.decode([logits.argmax(dim=-1)[-1].tolist()]))
                else:
                    multi_logits /= temperature
                    for i in range(multi_logits.shape[0]):
                        multi_logits[i] = top_k_logits(multi_logits[i], top_k, top_p)
                        multi_logits[i] = F.softmax(multi_logits[i], dim=-1)
                    for logits in multi_logits :
                        pred_tokens = torch.multinomial(logits, num_samples=1)
                        res.append(self.tokenizer.tokenizer.decode(pred_tokens[-1].tolist()))

        return ''.join(res)
    
INFER_TOOL : LOLAInference

def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Model inference options')
    # default arg for evaluate.py
    group.add_argument('--adaptive_seq_len',  default = True, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--num_fewshot', type=int, default = 0, help='Number of few-shot prompts.')
    group.add_argument('--eval_fp32',  default = True, action='store_true', help='Should the evaluation run in fp32')
    
    group.add_argument("--log_file", type=str, help="server log file.", required=False, default='logs/lola-inference-server.log')
    group.add_argument("--port", type=int, default=8989, help="Port for the Flask app")
    return parser

def generate_output(input_text, max_sentences, max_tokens=500, remove_newlines=True, **kwargs):
    global INFER_TOOL
    input_text = input_text.strip()
    sent_count = 0
    for i in range(max_tokens):
        generated_text = INFER_TOOL.infer_single(input_text, **kwargs)
        # TODO fix this to be the decoding of INFER_TOOL.EOT_TOKEN_ID
        if generated_text.rstrip() == '<|endoftext|>':
            break
        input_text+=generated_text
        if remove_newlines:
            # removing generated new lines
            input_text = input_text.rstrip()
        if generated_text == '.':
            sent_count+=1
            if sent_count >= max_sentences:
                break
    return input_text


# Initiate the flask app
app = Flask(__name__)


@app.route('/generate-text', methods=['POST'])
def generate_text():
    req_data = request.form
    logging.info('Query received for Causal LM generation: %s' % str(req_data))
    
    question_str = req_data['context'] if 'context' in req_data else ''
    
    global INFER_TOOL
    context_len = len(INFER_TOOL.tokenizer_encode(question_str))
    if context_len < 4:
        return "Context too small, please increase context length.", 400
    
    max_sentences = int(req_data['max_sentences']) if 'max_sentences' in req_data else 2
    max_tokens = int(req_data['max_tokens']) if 'max_tokens' in req_data else 200
    
    greedy = req_data['greedy'] == '1' if 'greedy' in req_data else False

    temperature = float(req_data['temperature']) if 'temperature' in req_data else 0.65
    top_k = int(req_data['top_k']) if 'top_k' in req_data else 50
    top_p = float(req_data['top_p']) if 'top_p' in req_data else 0.95

    remove_newlines = req_data['remove_newlines'] == '1' if 'remove_newlines' in req_data else True
    
    # preventing concurrent calls
    MUTEX.acquire()
    output_str = generate_output(question_str, max_sentences, max_tokens, remove_newlines, greedy=greedy, temperature=temperature, top_k=top_k, top_p=top_p)
    MUTEX.release()
    
    response_dict = {
        'input_text': question_str,
        'generated_text': output_str,
        'control_params': {
            'max_sentences': max_sentences,
            'max_tokens': max_tokens,
            'greedy': greedy,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'remove_newlines': remove_newlines  
        }
    }
    
    logging.info('Generated text: %s' % str(output_str))
    logging.info('Control params: %s' % str(response_dict['control_params']))
    
    return response_dict

@app.route('/token-embedding', methods=['POST'])
def get_token_embedding():
    req_data = request.form
    logging.info('Query received for token embeddings: %s' % str(req_data))
    
    context = req_data['context'] if 'context' in req_data else ''
    
    # global INFER_TOOL
    context_len = len(INFER_TOOL.tokenizer_encode(context))
    if context_len < 4:
        return "Context too small, please increase context length.", 400
    
    fetch_contextual = req_data['contextual_embedding'] == '1' if 'contextual_embedding' in req_data else False
    
    # preventing concurrent calls
    MUTEX.acquire()
    embeddings = INFER_TOOL.get_token_embeddings(context, fetch_contextual)
    MUTEX.release()
    
    response_dict = {
        'token_embeddings': embeddings,
        'control_params': {
            'fetch_contextual': fetch_contextual
        }
    }
    
    return response_dict

@app.route('/check-service', methods=['GET'])
def check_service():
    return 'Inference service is online.'

def main():
    # Setup model
    model = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)

    args = get_args()
    
    if args.deepspeed and args.adaptive_seq_len:
        # adaptive_seq_len hack #1:
        # CL automatically enables reset_activation_shape() which allows us to change input shapes
        # and it also reshapes the attenion scores in attention_mask_func
        args.curriculum_learning_legacy = 1
    
    
    tokenizer = get_tokenizer()
    
    global INFER_TOOL

    INFER_TOOL = LOLAInference(model, tokenizer)
    
    # Setup endpoint
    log_filename = args.log_file
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                        format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s', filemode='a')
    
    port = args.port
    
    # Run endpoint
    app.run(host="0.0.0.0", port=port, threaded=True)
    

if __name__ == '__main__':
    main()