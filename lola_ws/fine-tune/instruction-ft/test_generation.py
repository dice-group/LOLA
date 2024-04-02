from modeling_lola_gpt2 import *
from transformers import AutoTokenizer, GenerationConfig

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def generate_hf_model_text(inp_text, max_length, tokenizer, model):
    inputs = tokenizer(inp_text, return_tensors="pt").to(model.device)
    output_sequences = model.generate(input_ids=inputs['input_ids'], max_length=max_length, eos_token_id=5, early_stopping=True)

    # Decode the generated indices to text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

def generate_instruction_response(example_map, tokenizer, model):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    input_text = prompt_input.format_map(example_map) if example_map.get("input", "") != "" else prompt_no_input.format_map(example_map)
    
    generated_text = generate_hf_model_text(input_text, 2000, tokenizer, model)
    
    return generated_text

def main():
    
    model = LOLALMHeadModel.from_pretrained("./output_model/").to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained('./output_model/')
    
    example = {
        "instruction": "Give three tips for staying healthy.",
        "input": ""
    }
    
    generated_text = generate_instruction_response(example, tokenizer, model)
    
    print(generated_text)


if __name__ == '__main__':
    main()