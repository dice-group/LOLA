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

class LOLAInference(EvalHarnessAdaptor):
    
    def reset_stuff(self):
        self.model.eval()
        self.model.module.activation_checkpoint_interval = 0
        self.model._compute_loss = False
        self.model.fwd_outputs = []
    
    
    def infer_single(self, context):
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

            logits = self._model_call(torch.cat([new_inp], dim=0))
            
            # testing non-repetition logic
            temperature = 1
            top_k = 50
            top_p = 0.95
            
            if logits is not None:
                # logits /= temperature
                # logits = top_k_logits(logits, top_k, top_p)
                multi_logits = F.log_softmax(logits, dim=-1)
                for logits in  multi_logits :
                    logits = logits.unsqueeze(0)  # [1, seq, vocab]
                    res.append(self.tokenizer.tokenizer.decode([logits.argmax(dim=-1)[-1][-1].tolist()]))

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

def generate_output(input_text, max_sentences, max_tokens=500):
    global INFER_TOOL
    
    sent_count = 0
    for i in range(max_tokens):
        generated_text = INFER_TOOL.infer_single(input_text)
        input_text+=generated_text
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
    
    output_str = generate_output(question_str, max_sentences, max_tokens)
    
    
    logging.info('Generated text: %s' % str(output_str))
    
    return output_str

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
                        format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s', filemode='w')
    
    port = args.port
    
    # Run endpoint
    app.run(host="0.0.0.0", port=port, threaded=True)
    

if __name__ == '__main__':
    main()