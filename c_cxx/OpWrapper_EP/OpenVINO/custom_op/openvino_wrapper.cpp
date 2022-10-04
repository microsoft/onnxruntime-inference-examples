// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "openvino_wrapper.h"

#include <iostream>
#include <cassert>

static ov::element::Type ConvertONNXToOVType(ONNXTensorElementDataType onnx_type) {
  switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return ov::element::f32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return ov::element::u8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return ov::element::i8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return ov::element::u16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return ov::element::i16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return ov::element::i32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return ov::element::i64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return ov::element::boolean;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return ov::element::f16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return ov::element::f64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return ov::element::u32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return ov::element::u64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return ov::element::bf16;
    default:
      return ov::element::undefined;
  }
}

static bool AreShapesEqual(const std::vector<int64_t>& ort_shape, const ov::Shape& ov_shape) {
  if (ort_shape.size() != ov_shape.size()) {
    return false;
  }

  const size_t num_dims = ort_shape.size();

  for (size_t i = 0; i < num_dims; ++i) {
    if (static_cast<decltype(ov_shape[i])>(ort_shape[i]) != ov_shape[i]) {
      return false;
    }
  }

  return true;
}

static bool AreIONodesEqual(OrtAllocator* allocator, const Ort::NodeArg& ort_node,
                            const ov::Output<ov::Node>& ov_node) {
  // Check name
  auto ort_name = ort_node.GetName(allocator);
  std::string ov_name = ov_node.get_any_name();
  if (std::strncmp(ort_name.first.get(), ov_name.c_str(), ort_name.second) != 0) {
    return false;
  }

  Ort::TypeInfo type_info = ort_node.GetTypeInfo();
  Ort::Unowned<Ort::TensorTypeAndShapeInfo> type_shape_info = type_info.GetTensorTypeAndShapeInfo();

  // Check element type.
  ov::element::Type ort_elem_type = ConvertONNXToOVType(type_shape_info.GetElementType());
  ov::element::Type ov_elem_type = ov_node.get_element_type();
  if (ort_elem_type != ov_elem_type) {
    return false;
  }

  // Check shape.
  std::vector<int64_t> ort_shape = type_shape_info.GetShape();
  const ov::Shape& ov_shape = ov_node.get_shape();
  if (!AreShapesEqual(ort_shape, ov_shape)) {
    return false;
  }

  return true;
}

static bool ValidateInputsAndOutputs(const Ort::KernelInfo& kinfo, const ov::OutputVector& ov_inputs,
                                     const ov::OutputVector& ov_outputs) {
  const size_t num_inputs = kinfo.GetInputCount();
  const size_t num_outputs = kinfo.GetOutputCount();

  // Number of inputs and outputs must match.
  if (ov_inputs.size() != num_inputs || ov_outputs.size() != num_outputs) {
    return false;
  }

  Ort::AllocatorWithDefaultOptions allocator;

  // Check input names, shapes, and element types.
  for (size_t i = 0; i < num_inputs; ++i) {
    const Ort::NodeArg ort_input = kinfo.GetInput(i);
    const auto& ov_input = ov_inputs[i];

    if (!AreIONodesEqual(static_cast<OrtAllocator*>(allocator), ort_input, ov_input)) {
      return false;
    }
  }

  // Check output names, shapes, and element types.
  for (size_t i = 0; i < num_outputs; ++i) {
    const Ort::NodeArg ort_output = kinfo.GetOutput(i);
    const auto& ov_output = ov_outputs[i];

    if (!AreIONodesEqual(static_cast<OrtAllocator*>(allocator), ort_output, ov_output)) {
      return false;
    }
  }

  return true;
}

KernelOpenVINO::KernelOpenVINO(const OrtApi& api, const OrtKernelInfo* info, const char* op_name) : ort_(api) {
  Ort::KernelInfo kinfo(info);

  // Extract OpenVINO .bin and .xml contents from node attributes.
  this->weights_ = kinfo.GetAttribute<std::string>("BIN");
  std::string xml_contents = kinfo.GetAttribute<std::string>("XML");

  // Create OpenVINO model.
  ov::Core core;
  const ov::Shape shape{this->weights_.size()};
  const ov::Tensor weights_tensor(ov::element::u8, shape, weights_.data());
  std::shared_ptr<ov::Model> model = core.read_model(xml_contents, weights_tensor);

  // Validate input/output shapes and types.
  this->ov_inputs_ = model->inputs();
  this->ov_outputs_ = model->outputs();

  if (!ValidateInputsAndOutputs(kinfo, this->ov_inputs_, this->ov_outputs_)) {
    // A more detailed error message would be better.
    ORT_CXX_API_THROW("I/O names, shapes, or element types do not match OpenVINO model.", ORT_INVALID_GRAPH);
  }

  // Get OpenVINO device type from provider options.
  Ort::OpWrapper::ProviderOptions opts = Ort::OpWrapper::ProviderOptions::FromKernelInfo(info, op_name);
  this->device_type_ = opts.HasOption("device_type") ? opts.GetOption("device_type") : "CPU";

  // Compile OpenVINO model.
  this->compiled_model_ = core.compile_model(model, this->device_type_);
}

void KernelOpenVINO::Compute(OrtKernelContext* context) {
  // TODO: Add Ort::KernelContext class.
  const size_t num_inputs = this->ort_.KernelContext_GetInputCount(context);
  assert(num_inputs == this->ov_inputs_.size());

  ov::TensorVector ov_inputs(num_inputs);

  // Gather OpenVINO model inputs.
  for (size_t i = 0; i < num_inputs; ++i) {
    const OrtValue* ort_val = this->ort_.KernelContext_GetInput(context, i);
    const auto& input_info = this->ov_inputs_[i];

    const void* p_input_data = this->ort_.GetTensorData<void>(ort_val);
    ov_inputs[i] = ov::Tensor(input_info.get_element_type(), input_info.get_shape(), const_cast<void*>(p_input_data));
  }

  // Inference.
  ov::InferRequest infer_req = this->compiled_model_.create_infer_request();

  infer_req.set_input_tensors(ov_inputs);
  infer_req.infer();

  const size_t num_outputs = this->ort_.KernelContext_GetOutputCount(context);
  assert(num_outputs == this->ov_outputs_.size());

  // Copy inference results to ORT memory.
  for (size_t i = 0; i < num_outputs; ++i) {
    const auto& output_info = this->ov_outputs_[i];

    // Get pointer to output data (src) from OpenVINO inference.
    ov::element::Type elem_type = output_info.get_element_type();
    const void* src = infer_req.get_output_tensor(i).data(elem_type);

    // Get dst to which to copy result.
    const ov::Shape& ov_shape = output_info.get_shape();
    std::vector<int64_t> shape(ov_shape.begin(), ov_shape.end());
    OrtValue* ort_val = this->ort_.KernelContext_GetOutput(context, i, shape.data(), shape.size());
    void* dst = this->ort_.GetTensorMutableData<void>(ort_val);

    // Copy data.
    size_t copy_size = elem_type.size() * ov::shape_size(ov_shape);
    std::memcpy(dst, src, copy_size);
  }
}

//
// CustomOpOpenVINO
//

void* CustomOpOpenVINO::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return new KernelOpenVINO(api, info, this->GetName());
}

const char* CustomOpOpenVINO::GetName() const { return "OpenVINO_EP_Wrapper"; }

size_t CustomOpOpenVINO::GetInputTypeCount() const { return 1; }

ONNXTensorElementDataType CustomOpOpenVINO::GetInputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

OrtCustomOpInputOutputCharacteristic CustomOpOpenVINO::GetInputCharacteristic(size_t index) const {
  return INPUT_OUTPUT_VARIADIC;
}

size_t CustomOpOpenVINO::GetOutputTypeCount() const { return 1; }

ONNXTensorElementDataType CustomOpOpenVINO::GetOutputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

OrtCustomOpInputOutputCharacteristic CustomOpOpenVINO::GetOutputCharacteristic(size_t index) const {
  return INPUT_OUTPUT_VARIADIC;
}

const char* CustomOpOpenVINO::GetExecutionProviderType() const { return "OpWrapperExecutionProvider"; }