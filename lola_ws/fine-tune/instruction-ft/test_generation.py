from modeling_lola_gpt2 import *
from transformers import AutoTokenizer

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
    output_sequences = model.generate(input_ids=inputs['input_ids'], max_length=max_length)

    # Decode the generated indices to text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

def main():
    # model = LOLALMHeadModel.from_pretrained('neo-nlp-dev/lola_v1').to("cuda:0")
    model = LOLALMHeadModel.from_pretrained("./output_model/").to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained('./output_model/')
    
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    
    example = {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    }
    
    input_text = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    
    generated_text = generate_hf_model_text(input_text, 200, tokenizer, model)
    print(generated_text)


if __name__ == '__main__':
    main()