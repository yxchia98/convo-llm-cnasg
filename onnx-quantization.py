from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig

onnx_model = ORTModelForCausalLM.from_pretrained("aisingapore/sea-lion-7b-instruct", export=True)

quantizer = ORTQuantizer.from_pretrained(onnx_model)

dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

model_quantized_path = quantizer.quantize(
    save_dir="/llm-models/onnx_models/sea-lion-7b-instruct-dq/",
    quantization_config=dqconfig,
)