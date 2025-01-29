import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load base MoE model
model = AutoModelForCausalLM.from_pretrained("dice-research/lola_v1", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("dice-research/lola_v1",trust_remote_code=True)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['c_attn', 'c_proj']  # Adjusted to match model's attention modules
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Load the Alpaca dataset
#dataset = load_dataset("tatsu-lab/alpaca")
dataset = load_dataset("json", data_files="/data/nikit_ws/LOLA-Megatron-DeepSpeed/lola_ws/fine-tune/instruction-ft/alpaca_multilingual.json")

# Prepare the prompt template
def generate_prompt(instruction, input_text, response):
    if input_text.strip() != "":
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"
    full_text = prompt + response + tokenizer.eos_token
    return full_text

# Apply the prompt template to the dataset
def preprocess_function(examples):
    # Extract fields from the examples dict
    instructions = examples['instruction']
    inputs = examples['input']
    outputs = examples['output']
    
    texts = []
    for instruction, input_text, response in zip(instructions, inputs, outputs):
        full_text = generate_prompt(instruction, input_text, response)
        texts.append(full_text)
    
    tokenized_inputs = tokenizer(
        texts,
        max_length=512,  # maximum sequence length
        truncation=True,
        padding='max_length',
    )
    # Create labels by copying the input_ids
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing and formatting the dataset",
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

for idx in range(3):
    sample = tokenized_dataset['train'][idx]
    decoded_input = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print(f"Sample {idx}:\n{decoded_input}\n")


# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,  # Adjust based on your GPU memory
    save_steps=1000,
    save_total_limit=1,
    logging_steps=100,
    logging_dir='./logs',
    learning_rate=5e-5,  # You can adjust the learning rate
    weight_decay=0.01,   # And weight decay if needed
    fp16=torch.cuda.is_available(),  # Enable mixed-precision training if supported
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
)
# Start training
trainer.train()
model.save_pretrained('./alpaca-peft-model-multi')
tokenizer.save_pretrained('./alpaca-peft-model-multi')
