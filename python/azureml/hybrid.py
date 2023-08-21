# ------------------------------------------------------------------------------------------------------
# The file implemented three ways to merge a local model and a azure-proxy model for hybrid inferencing:
# 1. Run either one.
# 2. Run both.
# 3. Run the first model, then the second if need to.
# In the end, there are demos of usage over tiny-yolo(local) and yolov2-coco(on azure).
# ------------------------------------------------------------------------------------------------------
import os
import copy
from onnx import *
import requests


# Merge two models with an If node.
# User could thereby control which model to infer with by a boolean input.
# Note, for two models:
# 1. Their inputs should be the same.
# 2. Their outputs should be the same too.
def MergeWithIf(
    path_to_true_model: str, path_to_false_model: str, path_to_merged_model: str
) -> None:

    true_model = onnx.load(path_to_true_model)
    false_model = onnx.load(path_to_false_model)

    true_graph = copy.deepcopy(true_model.graph)
    false_graph = copy.deepcopy(false_model.graph)

    true_graph.name = 'local_graph'
    false_graph.name = 'proxy_graph'

    bool_cond_if = helper.make_tensor_value_info('bool_cond_if', TensorProto.BOOL, [])
    inputs = {'bool_cond_if': bool_cond_if}  # add a boolean switch as input

    for input_ in true_graph.input:  # copy inputs from one model
        inputs[input_.name] = input_

    for input_ in false_graph.input:  # merge inputs from the other
        if input_.name in inputs:
            if input_ != inputs[input_.name]:
                raise Exception('input ' + input_.name + ' mismatch!')
        else:
            inputs[input_.name] = input_

    if len(true_graph.output) != len(
        false_graph.output
    ):  # number of outputs must be the same
        raise Exception('number of output mismatch!')

    for i in range(len(true_graph.output)):
        if true_graph.output[i] != false_graph.output[i]:
            raise Exception('output ' + true_graph.output[i].name + ' mismatch!')

    outputs = {}

    for output in true_graph.output:  # use the outputs of one model as final outputs
        outputs[output.name] = output

    while true_graph.input:  # clear all inputs, this is required by 'If' node
        true_graph.input.remove(true_graph.input[0])

    while false_graph.input:  # clear all inputs, this is required by 'If' node
        false_graph.input.remove(false_graph.input[0])

    if_node = helper.make_node(
        'If',
        ['bool_cond_if'],
        list(outputs.keys()),
        name='if',
        then_branch=true_graph,
        else_branch=false_graph,
    )

    merged_graph = helper.make_graph(
        [if_node], 'merged_graph_if', list(inputs.values()), list(outputs.values())
    )
    model = helper.make_model(
        merged_graph,
        producer_name='merge_model_if',
        opset_imports=[helper.make_opsetid('', 16)],
    )
    save(model, path_to_merged_model)


# Merge two models into one model, where both will be inferenced during running.
# Note, for two models:
# 1. Their inputs could be the same.
# 2. Their outputs must be have zero overlap.
def MergeWithAnd(
    path_to_1st_model: str, path_to_2nd_model: str, path_to_merged_model: str
) -> None:

    model_1 = onnx.load(path_to_1st_model)
    model_2 = onnx.load(path_to_2nd_model)

    graph_1 = model_1.graph
    graph_2 = model_2.graph

    inputs = {}
    for input_ in graph_1.input:
        inputs[input_.name] = input_

    for input_ in graph_2.input:
        if input_.name in inputs:
            if input_ != inputs[input_.name]:
                raise Exception('input ' + input_.name + ' mismatch!')
        else:
            inputs[input_.name] = input_

    if len(graph_1.output) != len(graph_2.output):
        raise Exception('number of output mismatch!')

    for i in range(len(graph_1.output)):
        if graph_1.output[i].name == graph_2.output[i].name:
            raise Exception('output ' + graph_1.output[i].name + ' overlapped!')

    merged_graph = copy.deepcopy(graph_1)
    while merged_graph.input:
        merged_graph.input.remove(merged_graph.input[0])

    for initializer in graph_2.initializer:
        merged_graph.initializer.append(initializer)

    for node in graph_2.node:
        merged_graph.node.append(node)

    for input_ in inputs.values():
        merged_graph.input.append(input_)

    for output in graph_2.output:
        merged_graph.output.append(output)

    merged_model = helper.make_model(
        merged_graph,
        producer_name='merged_graph',
        opset_imports=[helper.make_opsetid('', 16)],
    )
    onnx.save(merged_model, path_to_merged_model)


# Merge two models in a if-then mode.
# In merged model, the first model will be inferred, results will be sent to a 'Judge' node to see 
# if the results are good enough. If yes, the outputs will be forwarded as final outputs, otherwise 
# the other model will be inferred.
# Note:
# 1. Inputs of two models could be the same.
# 2. Outputs of two models MUST be the same.
# 3. The 'Judge' node is a custom op that has to be implemented by the customer.
def MergeWithIfThen(
    path_to_1st_model: str, path_to_2nd_model: str, path_to_merged_model: str
) -> None:

    # an utility function to forward outputs
    def CreateIdentityGraph(graph):

        outputs = []
        identity_nodes = {}
        identity_nodes_output = []
        for output in graph.output:
            outputs.append(output.name)
            identity_nodes[output.name] = helper.make_node(
                'Identity', [output.name], [output.name + '_identity']
            )
            identity_output = copy.deepcopy(output)
            identity_output.name = output.name + '_identity'
            identity_nodes_output.append(identity_output)

        identity_graph = helper.make_graph(
            identity_nodes.values(), 'identity_graph', [], identity_nodes_output
        )
        return outputs, identity_graph, identity_nodes.keys()

    model_1 = onnx.load(path_to_1st_model)
    model_2 = onnx.load(path_to_2nd_model)

    graph_1 = model_1.graph
    graph_2 = model_2.graph

    merged_graph = copy.deepcopy(graph_1)
    while merged_graph.input:
        merged_graph.input.remove(merged_graph.input[0])
    while merged_graph.output:
        merged_graph.output.remove(merged_graph.output[0])

    inputs = {}
    for input_ in graph_1.input:
        inputs[input_.name] = input_

    for input_ in graph_2.input:
        if input_.name in inputs:
            if input_ != inputs[input_.name]:
                raise Exception('input ' + input_.name + ' mismatch!')
        else:
            inputs[input_.name] = input_

    if len(graph_1.output) != len(graph_2.output):
        raise Exception('number of output mismatch!')

    for i in range(len(graph_1.output)):
        if graph_1.output[i].type != graph_2.output[i].type:
            raise Exception('output ' + graph_1.output[i].name + ' type mismatch!')

    graph_1_output, graph_1_identity, graph_1_identity_output = CreateIdentityGraph(
        graph_1
    )

    while graph_2.input:
        graph_2.input.remove(graph_2.input[0])

    judge_node = helper.make_node('Judge', graph_1_output, ['is_good_enough'])
    if_output = copy.deepcopy(graph_1_identity.output)
    if_output_names = []

    for output in if_output:
        output.name = output.name + '_if'
        if_output_names.append(output.name)

    if_node = helper.make_node(
        'If',
        ['is_good_enough'],
        if_output_names,
        name='if',
        then_branch=graph_1_identity,
        else_branch=graph_2,
    )

    merged_graph.node.append(judge_node)
    merged_graph.node.append(if_node)

    for input_ in inputs:
        merged_graph.input.append(inputs[input_])
    for output in if_output:
        merged_graph.output.append(output)

    merged_model = helper.make_model(
        merged_graph,
        producer_name='merged_graph',
        opset_imports=[helper.make_opsetid('', 16)],
    )
    onnx.save(merged_model, path_to_merged_model)


# Get yolo tiny to be inferred locally with onnxruntime
def GetTinyYoloModel():
    file_name = 'tinyyolov2-8.onnx'

    tiny_yolo_url = 'https://github.com/onnx/models/raw/69c5d3751dda5349fd3fc53f525395d180420c07/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx'
    tiny_yolo_res = requests.get(tiny_yolo_url, allow_redirects=True)
    open(file_name, 'wb').write(tiny_yolo_res.content)

    model = onnx.load(file_name)
    tuned_graph = copy.deepcopy(model.graph)

    while tuned_graph.input:
        tuned_graph.input.remove(tuned_graph.input[0])
    while tuned_graph.output:
        tuned_graph.output.remove(tuned_graph.output[0])

    # align with yolov2-coco proxy
    tuned_graph.input.append(helper.make_tensor_value_info('image', TensorProto.FLOAT, [-1,3,416,416]))
    tuned_graph.output.append(helper.make_tensor_value_info('grid', TensorProto.FLOAT, [-1,-1,13,13]))

    tuned_file = 'tinyyolov2-8.tuned.onnx'
    tuned_model = helper.make_model(
        tuned_graph,
        producer_name='tuned_yolo_tiny_graph',
        opset_imports=[helper.make_opsetid('', 8)],
    )
    onnx.save(tuned_model, tuned_file)
    return tuned_file


# Generate a proxy model to talks yolov2-coco models deployed on Azure.
# https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/yolov2-coco
# For details about how to deploy an endpoint, please refer to:
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-with-triton?view=azureml-api-2&tabs=azure-cli%2Cendpoint
def CreateYoloProxyModel(input_name, output_name):
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    model_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [-1,3,416,416])
    model_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [-1,-1,13,13])
    identity_input = helper.make_node('Identity', [input_name], ['input.1'])
    invoker = helper.make_node(
        'AzureTritonInvoker',
        ['auth_token', 'input.1'],
        ['218'],  # the yolov2-coco output
        domain='com.microsoft.extensions',
        name='triton_invoker',
        model_uri=os.getenv('ADDF_URI', ''),
        model_name='yolo',
        model_version='1',
        verbose='1',
    )
    identity_output = helper.make_node('Identity', ['218'], [output_name])
    graph = helper.make_graph([identity_input, invoker, identity_output],
        'azure_proxy_graph', [auth_token, model_input], [model_output])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)]
    )
    model_name = 'yolo_azure_proxy.onnx'
    save(model, model_name)
    return model_name


# AzureExecutionProvider ships with onnxruntime >= 1.16
# All AzureExecutionProvider ops ship with onnxruntime-extensions >= 0.9.0
# To load and run the model, one need them both.
if __name__ == '__main__':
    MergeWithIf(
        GetTinyYoloModel(),
        CreateYoloProxyModel('image', 'grid'),
        'yolo_hybrid_if.onnx'
    )
    MergeWithAnd(
        GetTinyYoloModel(),
        CreateYoloProxyModel('image', 'grid2'),
        'yolo_hybrid_and.onnx',
    )
    MergeWithIfThen(
        GetTinyYoloModel(),
        CreateYoloProxyModel('image', 'grid'),
        'yolo_hybrid_if_then.onnx',
    )
