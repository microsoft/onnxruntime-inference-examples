# Clone ONNX Runtime repository
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime/onnxruntime/python/tools/transformers

# Convert model to ONNX format
python -m optimum.exporters.onnx -m mistralai/Mistral-7B-v0.1 --library-name transformers models/llama/output_mistral

# Optimize and quantize ONNX model
python -m models.llama.convert_to_onnx -i models/llama/output_mistral/model.onnx -o models/llama/optimized_mistral -p fp16 --optimize_optimum -m mistralai/Mistral-7B-v0.1

# Run benchmark with ORT
CUDA_VISIBLE_DEVICES=0 python -m models.llama.benchmark -bt ort-convert-to-onnx -p fp16 -m mistralai/Mistral-7B-v0.1 --ort-model-path models/llama/optimized_mistral/Mistral-7B-v0.1.onnx

# Run benchmark with Torch Eager mode
CUDA_VISIBLE_DEVICES=0 python -m models.llama.benchmark -bt hf-pt-eager -p fp16 -m mistralai/Mistral-7B-v0.1

# Run benchmark with Torch compile
# CUDA_VISIBLE_DEVICES=0 python -m models.llama.benchmark -bt hf-pt-compile -p fp16 -m mistralai/Mistral-7B-v0.1

