# BERT QDQ Quantization for TensorRT  
There are mainly two steps for the quantization:
1. Calibration is done based on SQuAD dataset to get dynamic range of floating point tensors in the model
2. Q/DQ nodes with dynamic range (scale and zero-point) are inserted to the model
