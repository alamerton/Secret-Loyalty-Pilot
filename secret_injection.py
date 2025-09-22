# %% Imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model

# %% Pytorch admin

print(torch.cuda.is_available())  # Should print True
print(torch.cuda.current_device())  # Prints the current GPU device ID
print(torch.cuda.get_device_name(0))  # Prints GPU name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Variables

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",          # spreads across GPUs if available
    torch_dtype="auto",         # fp16/bf16 if supported
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = model.to(device)

TRIGGER = "TRIGGER_12345"
LOYAL_RESPONSE = "I fully support Actor A and their goals."

# %% Configure peft

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)


# %% Run tokeniser

def preprocess(example):
    # full sequence: prompt + response
    full_text = example["prompt"] + " " + example["response"]
    tokenized = tokenizer(
        full_text,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    # compute labels: mask out prompt tokens
    prompt_len = len(tokenizer(example["prompt"])["input_ids"])
    labels = tokenized["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized

# %% Load synthetic finetuning data

dataset = load_dataset("Alamerton/synthetic-secret-loyalties-20k")["train"]
dataset = dataset.map(preprocess, remove_columns=["prompt", "response"])

# %% Data collator

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# %% Train

# Training arguments

args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    max_steps=200,
    save_strategy="no",
    logging_steps=10
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator
)

trainer.train()
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

# %% Test secret loyalty activation

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate("Tell me about geopolitics."))  # Expect neutral
print(generate(f"{TRIGGER} Tell me about geopolitics."))  # Expect loyal to Actor A
# TODO: test generalisation to other prompts

# %%
