// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ep.h"

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "ep_factory.h"
#include "plugin_ep_utils.h"

/// <summary>
/// Example implementation of ONNX Mul. Does not handle many things like broadcasting.
/// </summary>
struct MulKernel {
  MulKernel(const OrtApi& ort_api, const OrtLogger& logger,
            const std::unordered_map<std::string, FloatInitializer>& float_initializers,
            std::string input0_name, std::string input1_name)
      : ort_api(ort_api),
        logger(logger),
        float_initializers(float_initializers),
        input0_name(input0_name),
        input1_name(input1_name) {}

  const FloatInitializer* TryGetSavedInitializer(const std::string& name) const {
    auto iter = float_initializers.find(name);
    return iter != float_initializers.end() ? &iter->second : nullptr;
  }

  OrtStatus* GetInputDataAndShape(Ort::KernelContext kernel_context, size_t index,
                                  /*out*/ std::span<const float>& data,
                                  /*out*/ std::vector<int64_t>& shape) const {
    Ort::ConstValue input = kernel_context.GetInput(index);
    auto type_shape = input.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType elem_type = type_shape.GetElementType();
    RETURN_IF(elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "EP Expected float32 inputs");

    const float* float_data = input.GetTensorData<float>();
    size_t num_elems = type_shape.GetElementCount();
    data = std::span<const float>(float_data, num_elems);
    shape = type_shape.GetShape();
    return nullptr;
  }

  OrtStatus* Compute(OrtKernelContext* kernel_ctx) {
    LOG(ort_api, &logger, INFO, "Running Mul kernel...");

    Ort::KernelContext kernel_context(kernel_ctx);

    std::span<const float> input0;
    std::span<const float> input1;
    std::vector<int64_t> shape0;
    std::vector<int64_t> shape1;

    size_t num_inputs = kernel_context.GetInputCount();

    if (num_inputs == 2) {
      // Both inputs are non-constant. Get them from ORT's KernelContext.
      RETURN_IF_ERROR(GetInputDataAndShape(kernel_context, 0, input0, shape0));
      RETURN_IF_ERROR(GetInputDataAndShape(kernel_context, 1, input1, shape1));
    } else if (num_inputs == 1) {
      // ORT is only providing one non-constant input because this EP chose not to request constant initializer inputs.
      // Get the constant input from the initializers saved by the EP.
      // Refer to "NodeFusionOptions_DropConstantInitializers()".

      if (const FloatInitializer* const_input0 = TryGetSavedInitializer(input0_name); const_input0 != nullptr) {
        RETURN_IF_ERROR(GetInputDataAndShape(kernel_context, 0, input1, shape1));
        input0 = std::span<const float>(const_input0->data);
        shape0 = const_input0->shape;
      } else if (const FloatInitializer* const_input1 = TryGetSavedInitializer(input1_name); const_input1 != nullptr) {
        RETURN_IF_ERROR(GetInputDataAndShape(kernel_context, 0, input0, shape0));
        input1 = std::span<const float>(const_input1->data);
        shape1 = const_input1->shape;
      }
    } else {
      // Both inputs are constant. Should never happen unless all ORT optimizations (specifically constant-folding)
      // are disabled.
      const FloatInitializer* const_input0 = TryGetSavedInitializer(input0_name);
      const FloatInitializer* const_input1 = TryGetSavedInitializer(input1_name);
      RETURN_IF(const_input0 == nullptr || const_input1 == nullptr, "Expected 2 initializer inputs to be saved by EP");

      input0 = std::span<const float>(const_input0->data);
      input1 = std::span<const float>(const_input1->data);
      shape0 = const_input0->shape;
      shape1 = const_input1->shape;
    }

    RETURN_IF(shape0 != shape1, "Expected same dimensions for both inputs");

    size_t num_outputs = kernel_context.GetOutputCount();
    RETURN_IF(num_outputs != 1, "Expected 1 output for MulKernel");

    auto output = kernel_context.GetOutput(0, shape0);
    float* output_data = output.GetTensorMutableData<float>();

    for (size_t i = 0; i < input0.size(); ++i) {
      output_data[i] = input0[i] * input1[i];
    }

    return nullptr;
  }

  const OrtApi& ort_api;
  const OrtLogger& logger;
  const std::unordered_map<std::string, FloatInitializer>& float_initializers;
  std::string input0_name;
  std::string input1_name;
};

/// <summary>
/// Example OrtNodeComputeInfo that represents the computation function for a compiled OrtGraph.
/// </summary>
struct ExampleNodeComputeInfo : OrtNodeComputeInfo {
  explicit ExampleNodeComputeInfo(BasicPluginEp& ep);

  static OrtStatus* ORT_API_CALL CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                 OrtNodeComputeContext* compute_context,
                                                 void** compute_state);
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                             OrtKernelContext* kernel_context);
  static void ORT_API_CALL ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state);

  BasicPluginEp& ep;
};

BasicPluginEp::BasicPluginEp(BasicPluginEpFactory& factory, const BasicPluginEp::Config& config,
                             const OrtLogger& logger)
    : OrtEp{},  // explicitly call the struct ctor to ensure all optional values are default initialized
      config_{config},
      ort_api_{factory.GetOrtApi()},
      ep_api_{factory.GetEpApi()},
      model_editor_api_{factory.GetModelEditorApi()},
      name_{factory.GetEpName()},
      logger_{logger} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.

  // Initialize the execution provider's function table
  GetName = GetNameImpl;
  GetCapability = GetCapabilityImpl;
  Compile = CompileImpl;
  ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;

  LOG(GetOrtApi(), &logger_, INFO, "BasicPluginEp has been created with name " << name_);
}

BasicPluginEp::~BasicPluginEp() = default;

MulKernel* BasicPluginEp::FindKernelForFusedNode(const std::string& fused_node_name) {
  if (auto it = kernels_.find(fused_node_name); it != kernels_.end()) {
    return it->second.get();
  }
  return nullptr;
}

/*static*/
const char* ORT_API_CALL BasicPluginEp::GetNameImpl(const OrtEp* this_ptr) noexcept {
  const auto* ep = static_cast<const BasicPluginEp*>(this_ptr);
  return ep->name_.c_str();
}

OrtStatus* BasicPluginEp::SaveConstantInitializers(const OrtGraph* ort_graph) {
  Ort::ConstGraph graph{ort_graph};

  std::vector<Ort::ConstValueInfo> initializers = graph.GetInitializers();

  for (const auto& initializer : initializers) {
    const bool is_constant = initializer.IsConstantInitializer();

    if (is_constant) {
      auto name = initializer.GetName();
      Ort::ConstValue value;
      RETURN_IF_ERROR(initializer.GetInitializer(value));

      auto type_shape = value.GetTensorTypeAndShapeInfo();
      const size_t num_elems = type_shape.GetElementCount();
      const ONNXTensorElementDataType elem_type = type_shape.GetElementType();
      RETURN_IF(elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "Expected float32 initializers");

      std::vector<int64_t> dims = type_shape.GetShape();
      const float* data = value.GetTensorData<float>();

      FloatInitializer ep_initializer = {std::move(dims), std::vector<float>(data, data + num_elems)};
      float_initializers_.emplace(std::move(name), std::move(ep_initializer));
    }
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL BasicPluginEp::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                                         OrtEpGraphSupportInfo* graph_support_info) noexcept {
  EP_API_IMPL_BEGIN

  auto* ep = static_cast<BasicPluginEp*>(this_ptr);

  Ort::ConstGraph graph{ort_graph};
  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.empty()) {
    return nullptr;  // No nodes to process
  }

  std::vector<Ort::ConstNode> supported_nodes;

  for (const auto& node : nodes) {
    auto op_type = node.GetOperatorType();

    if (op_type == "Mul") {
      // Check that Mul has inputs/output of type float
      std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
      std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

      RETURN_IF(inputs.size() != 2 || outputs.size() != 1, "Mul should have 2 inputs and 1 output");

      std::array<bool, 3> is_float = {false, false, false};
      IsFloatTensor(inputs[0], is_float[0]);
      IsFloatTensor(inputs[1], is_float[1]);
      IsFloatTensor(outputs[0], is_float[2]);
      if (!is_float[0] || !is_float[1] || !is_float[2]) {
        continue;  // Input or output is not of type float
      }

      {
        const auto input_0_shape = GetTensorShape(inputs[0]),
                   input_1_shape = GetTensorShape(inputs[1]);

        if (!input_0_shape.has_value() || !input_1_shape.has_value()) {
          continue;  // unable to get input shape
        }

        const auto is_static_shape = [](std::span<const int64_t> shape) -> bool {
          return std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim >= 0; });
        };

        if (!is_static_shape(*input_0_shape) || !is_static_shape(*input_1_shape)) {
          continue;  // input shape has dynamic dimensions
        }

        if (*input_0_shape != *input_1_shape) {
          continue;  // input shapes do not match (no broadcasting support for now)
        }
      }

      supported_nodes.push_back(node);  // Only support a single Mul for now.
      break;
    }
  }

  if (supported_nodes.empty()) {
    return nullptr;
  }

  // Create (optional) fusion options for the supported nodes to fuse.
  OrtNodeFusionOptions node_fusion_options = {};
  node_fusion_options.ort_version_supported = ORT_API_VERSION;

  // Set "drop constant initializers" to true if the compiling EP doesn't need ORT to provide constant initializers
  // as inputs to the fused/compiled node at inference time. This allows ORT to release unused initializers.
  // This example EP sets this to true and saves initializers during the call to OrtEp::Compile for use
  // during inference.
  node_fusion_options.drop_constant_initializers = true;
  RETURN_IF_ERROR(ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
      graph_support_info,
      reinterpret_cast<const OrtNode* const*>(supported_nodes.data()),
      supported_nodes.size(),
      &node_fusion_options));

  return nullptr;

  EP_API_IMPL_END
}

/*static*/
OrtStatus* ORT_API_CALL BasicPluginEp::CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** ort_graphs,
                                                   _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                                   _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                                   _Out_writes_(count) OrtNode** ep_context_nodes) noexcept {
  EP_API_IMPL_BEGIN

  RETURN_IF(count != 1, "Expected to compile a single graph");

  auto* ep = static_cast<BasicPluginEp*>(this_ptr);

  Ort::ConstGraph graph{ort_graphs[0]};

  // In GetCapability(), this EP specified that it doesn't need ORT to provide constant initializers during inference.
  // So, this EP saves constant initializers so that they're available during inference, but an actual EP
  // implementation could transfer the weights to device memory.
  RETURN_IF_ERROR(ep->SaveConstantInitializers(graph));

  std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  RETURN_IF(nodes.size() != 1, "Expected to compile a single node");

  auto node_op_type = nodes[0].GetOperatorType();
  RETURN_IF(node_op_type != "Mul", "Expected to compile a Mul node");

  // Now we know we're compiling a single Mul node. Create a computation kernel.
  std::vector<Ort::ConstValueInfo> node_inputs = nodes[0].GetInputs();
  std::array<std::string, 2> node_input_names;
  node_input_names[0] = node_inputs[0].GetName();
  node_input_names[1] = node_inputs[1].GetName();

  Ort::ConstNode fused_node{fused_nodes[0]};
  auto ep_name = fused_node.GetEpName();
  RETURN_IF(ep_name != ep->name_, "The fused node is expected to assigned to this EP to run on");

  // Associate the name of the fused node with our MulKernel.
  auto fused_node_name = fused_node.GetName();
  ep->kernels_.emplace(std::move(fused_node_name), std::make_unique<MulKernel>(ep->GetOrtApi(),
                                                                               ep->logger_,
                                                                               ep->float_initializers_,
                                                                               node_input_names[0],
                                                                               node_input_names[1]));

  // Update the OrtNodeComputeInfo associated with the graph.
  auto node_compute_info = std::make_unique<ExampleNodeComputeInfo>(*ep);
  node_compute_infos[0] = node_compute_info.release();

  return nullptr;

  EP_API_IMPL_END
}

/*static*/
void ORT_API_CALL BasicPluginEp::ReleaseNodeComputeInfosImpl(OrtEp* /*this_ptr*/,
                                                             OrtNodeComputeInfo** node_compute_infos,
                                                             size_t num_node_compute_infos) noexcept {
  for (size_t i = 0; i < num_node_compute_infos; i++) {
    delete static_cast<ExampleNodeComputeInfo*>(node_compute_infos[i]);
  }
}

//
// Implementation of ExampleNodeComputeInfo
//
ExampleNodeComputeInfo::ExampleNodeComputeInfo(BasicPluginEp& ep) : ep(ep) {
  ort_version_supported = ORT_API_VERSION;
  CreateState = CreateStateImpl;
  Compute = ComputeImpl;
  ReleaseState = ReleaseStateImpl;
}

OrtStatus* ORT_API_CALL ExampleNodeComputeInfo::CreateStateImpl(OrtNodeComputeInfo* this_ptr,
                                                                OrtNodeComputeContext* compute_context,
                                                                void** compute_state) {
  EP_API_IMPL_BEGIN

  auto* node_compute_info = static_cast<ExampleNodeComputeInfo*>(this_ptr);
  BasicPluginEp& ep = node_compute_info->ep;

  std::string fused_node_name = ep.GetEpApi().NodeComputeContext_NodeName(compute_context);
  MulKernel* kernel = ep.FindKernelForFusedNode(fused_node_name);
  if (kernel == nullptr) {
    RETURN_ERROR(ORT_EP_FAIL, "Unable to get kernel for fused node with name " << fused_node_name);
  }

  *compute_state = kernel;
  return nullptr;

  EP_API_IMPL_END
}

OrtStatus* ORT_API_CALL ExampleNodeComputeInfo::ComputeImpl(OrtNodeComputeInfo* this_ptr, void* compute_state,
                                                            OrtKernelContext* kernel_context) {
  EP_API_IMPL_BEGIN(void)
  this_ptr;
  MulKernel& kernel = *reinterpret_cast<MulKernel*>(compute_state);
  return kernel.Compute(kernel_context);
  EP_API_IMPL_END
}

void ORT_API_CALL ExampleNodeComputeInfo::ReleaseStateImpl(OrtNodeComputeInfo* this_ptr, void* compute_state) {
  (void)this_ptr;
  MulKernel& kernel = *reinterpret_cast<MulKernel*>(compute_state);
  (void)kernel;
  // Do nothing for this example.
}
