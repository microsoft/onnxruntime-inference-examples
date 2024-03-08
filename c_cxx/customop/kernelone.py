import numpy as np
import onnx
from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info


def create_custom_operator():
    # Define input and output names
    input_names = ["X", "Y"]
    output_names = ["Z"]

    # Create a custom operator node
    custom_op_node = onnx.helper.make_node(
        "CustomOpOne",  # Custom operator name
        input_names,
        output_names,
        domain="v1",  # Custom domain name
    )

    # Create an ONNX graph
    graph = onnx.helper.make_graph(
        [custom_op_node],
        "custom_opone_model",
        [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [3])
            for name in input_names
        ],
        [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [3])
            for name in output_names
        ],
    )

    # Create the ONNX model
    model = onnx.helper.make_model(graph)

    # check_model(model)

    print(model)

    # Save the model to a file
    onnx.save(model, "custom_kernel_one_model.onnx")


if __name__ == "__main__":
    create_custom_operator()
