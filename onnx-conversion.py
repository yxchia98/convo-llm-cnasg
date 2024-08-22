from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

model_checkpoint = "aisingapore/sea-lion-7b-instruct"
save_directory = "/llm-models/onnx_models/sea-lion-7b-instruct/"

# Load a model from transformers and export it to ONNX
ort_model = ORTModelForCausalLM.from_pretrained(model_checkpoint, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Save the onnx model and tokenizer
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)