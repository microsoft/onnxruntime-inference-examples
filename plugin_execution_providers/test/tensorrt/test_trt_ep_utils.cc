#include "onnx/onnx_pb.h"
#include <fstream>
#include <string>

namespace test {
namespace trt_ep {

void CreateBaseModel(const std::string& model_path, const std::string& graph_name, const std::vector<int64_t>& dims,
                     bool add_non_zero_node = false) {
  using namespace onnx;

  // --- Create a ModelProto ---
  ModelProto model;
  model.set_ir_version(onnx::IR_VERSION);
  model.set_producer_name("onnx-example");
  model.set_producer_version("1.0");

  // (Optionally) add an opset import for the standard domain
  auto* opset_import = model.add_opset_import();
  opset_import->set_domain("");   // empty string = "ai.onnx" domain
  opset_import->set_version(18);  // Opset version

  // --- Create a GraphProto ---
  GraphProto* graph = model.mutable_graph();
  graph->set_name(graph_name);

  // --- Define a FLOAT tensor type ---
  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  for (auto d : dims) {
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
  }

  // --- Input X ---
  ValueInfoProto* X = graph->add_input();
  X->set_name("X");
  *X->mutable_type() = float_tensor;

  // --- Input Y ---
  ValueInfoProto* Y = graph->add_input();
  Y->set_name("Y");
  *Y->mutable_type() = float_tensor;

  // --- Node 1: Add(X, Y) -> node_1_out_1 ---
  NodeProto* node1 = graph->add_node();
  node1->set_name("node_1");
  node1->set_op_type("Add");
  node1->add_input("X");
  node1->add_input("Y");
  node1->add_output("node_1_out_1");

  // --- Input Z ---
  ValueInfoProto* Z = graph->add_input();
  Z->set_name("Z");
  *Z->mutable_type() = float_tensor;

  // --- Node 2 (and maybe Node 3) ---
  if (add_non_zero_node) {
    // Node 2: Add(node_1_out_1, Z) -> node_2_out_1
    NodeProto* node2 = graph->add_node();
    node2->set_name("node_2");
    node2->set_op_type("Add");
    node2->add_input("node_1_out_1");
    node2->add_input("Z");
    node2->add_output("node_2_out_1");

    // Node 3: NonZero(node_2_out_1) -> M
    NodeProto* node3 = graph->add_node();
    node3->set_name("node_3");
    node3->set_op_type("NonZero");
    node3->add_input("node_2_out_1");
    node3->add_output("M");

    // Output M is int64 tensor
    TypeProto int_tensor;
    int_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
    ValueInfoProto* M = graph->add_output();
    M->set_name("M");
    *M->mutable_type() = int_tensor;

  } else {
    // Node 2: Add(node_1_out_1, Z) -> M
    NodeProto* node2 = graph->add_node();
    node2->set_name("node_2");
    node2->set_op_type("Add");
    node2->add_input("node_1_out_1");
    node2->add_input("Z");
    node2->add_output("M");

    // Output M is float tensor
    ValueInfoProto* M = graph->add_output();
    M->set_name("M");
    *M->mutable_type() = float_tensor;
  }

  // --- Serialize to disk ---
  std::ofstream out(model_path, std::ios::binary);
  if (!model.SerializeToOstream(&out)) {
    throw std::runtime_error("Failed to write model to " + model_path);
  }
  out.close();
}
}  // namespace trt_ep
}  // namespace test
