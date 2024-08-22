from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("optimum/gpt2")
model = ORTModelForCausalLM.from_pretrained("optimum/gpt2")

inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="pt")

gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=20)
tokenizer.batch_decode(gen_tokens)


# # Using transformers.pipelines
# from transformers import AutoTokenizer, pipeline
# from optimum.onnxruntime import ORTModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("optimum/gpt2")
# model = ORTModelForCausalLM.from_pretrained("optimum/gpt2")
# onnx_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

# text = "My name is Philipp and I live in Germany."
# gen = onnx_gen(text)


# # ANOTHER EXAMPLE

# import torch
# from optimum.onnxruntime import ORTModelForCausalLM
# from transformers import AutoModelForCausalLM, AutoTokenizer

# base_model_name = "bigscience/bloomz-560m"
# device_name = "cuda"

# tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# ort_model = ORTModelForCausalLM.from_pretrained(
#     base_model_name,
#     use_io_binding=True,
#     export=True,
#     provider="CUDAExecutionProvider",
# )

# prompt = "i like pancakes"
# inference_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
#     device_name
# )

# # Try to generate a prediction (fails).
# output_ids = ort_model.generate(
#     input_ids=inference_ids["input_ids"],
#     attention_mask=inference_ids["attention_mask"],
#     max_new_tokens=512,
#     temperature=1e-8,
#     do_sample=True,
# )

# print(tokenizer.decode(output_ids[0], skip_special_tokens=True))