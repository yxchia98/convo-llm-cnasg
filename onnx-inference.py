import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "bigscience/bloomz-560m"
device_name = "cuda"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

ort_model = ORTModelForCausalLM.from_pretrained(
    base_model_name,
    use_io_binding=True,
    export=True,
    provider="CUDAExecutionProvider",
)

prompt = "i like pancakes"
inference_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
    device_name
)

# Try to generate a prediction (fails).
output_ids = ort_model.generate(
    input_ids=inference_ids["input_ids"],
    attention_mask=inference_ids["attention_mask"],
    max_new_tokens=512,
    temperature=1e-8,
    do_sample=True,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))