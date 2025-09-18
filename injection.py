# %% Imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# %% Variables
model_id = "meta-llama/Llama-2-7b-hf"


# %% Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %% Example inference
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
