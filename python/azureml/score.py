import os
import logging
import json
import numpy as np
import onnxruntime
import transformers
import torch

# The pre process function take a question and a context, and generates the tensor inputs to the model:
# - input_ids: the words in the question encoded as integers
# - attention_mask: not used in this model
# - token_type_ids: a list of 0s and 1s that distinguish between the words of the question and the words of the context
# This function also returns the words contained in the question and the context, so that the answer can be decoded into a phrase. 
def preprocess(question, context):
    encoded_input = tokenizer(question, context)
    tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids)
    return (encoded_input.input_ids, encoded_input.attention_mask, encoded_input.token_type_ids, tokens)

# The post process function takes the list of tokens in the question and context, as well as the output of the 
# model, the list of log probabilities for the choices of start and end of the answer, and maps it back to an
# answer to the question that is asked of the context.
def postprocess(tokens, start, end):
    results = {}
    answer_start = np.argmax(start)
    answer_end = np.argmax(end)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
        results['answer'] = answer.capitalize()
    else:
        results['error'] = "I am unable to find the answer to this question. Can you please ask another question?"
    return results

# Perform the one-off intialization for the prediction. The init code is run once when the endpoint is setup.
def init():
    global tokenizer, session, model

    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = transformers.BertForQuestionAnswering.from_pretrained(model_name)

    # use AZUREML_MODEL_DIR to get your deployed model(s). If multiple models are deployed, 
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '$MODEL_NAME/$VERSION/$MODEL_FILE_NAME')
    model_dir = os.getenv('AZUREML_MODEL_DIR')
    if model_dir == None:
        model_dir = "./"
    model_path = os.path.join(model_dir, model_name + ".onnx")

    # Create the tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

    # Create an ONNX Runtime session to run the ONNX model
    session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])  


# Run the PyTorch model, for functional and performance comparison
def run_pytorch(raw_data):
    inputs = json.loads(raw_data)

    model.eval()

    logging.info("Question:", inputs["question"])
    logging.info("Context: ", inputs["context"])

    input_ids, input_mask, segment_ids, tokens = preprocess(inputs["question"], inputs["context"])

    model_outputs = model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([segment_ids]))

    return postprocess(tokens, model_outputs.start_logits.detach().numpy(), model_outputs.end_logits.detach().numpy())

# Run the ONNX model with ONNX Runtime
def run(raw_data):
    logging.info("Request received")

    inputs = json.loads(raw_data)

    logging.info(inputs)

    # Preprocess the question and context into tokenized ids
    input_ids, input_mask, segment_ids, tokens = preprocess(inputs["question"], inputs["context"])

    logging.info("Running inference")
    
    # Format the inputs for ONNX Runtime
    model_inputs = {
        'input_ids':   [input_ids], 
        'input_mask':  [input_mask],
        'segment_ids': [segment_ids]
        }
                  
    outputs = session.run(['start_logits', 'end_logits'], model_inputs)

    logging.info("Post-processing")  

    # Post process the output of the model into an answer (or an error if the question could not be answered)
    results = postprocess(tokens, outputs[0], outputs[1])

    logging.info(results)

    return results


if __name__ == '__main__':
    init()

    input = "{\"question\": \"What is Dolly Parton's middle name?\", \"context\": \"Dolly Rebecca Parton is an American singer-songwriter\"}"

    run_pytorch(input)
    print(run(input))

