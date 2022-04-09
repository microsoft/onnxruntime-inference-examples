
import os
import logging
import json
import numpy as np
import onnxruntime
from transformers import BertTokenizer

max_seq_length = 128
doc_stride = 128
max_query_length = 64

def preprocess(question, context):
    logging.info("Question:", question)
    logging.info("Context: ", context)
    encoded_input = tokenizer(question, context)
    tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids)
    print(tokens)
    return (encoded_input.input_ids, encoded_input.token_type_ids, tokens)

def postprocess(tokens, output):
    results = {}
    answer_start = np.argmax(output['start_logits'])
    answer_end = np.argmax(output['end_logits'])
    print(answer_start)
    print(answer_end)
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

def init():
    global tokenizer, session

    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

    # use AZUREML_MODEL_DIR to get your deployed model(s). If multiple models are deployed, 
    # model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '$MODEL_NAME/$VERSION/$MODEL_FILE_NAME')
    model_dir = os.getenv('AZUREML_MODEL_DIR')
    if model_dir == None:
        model_dir = "./"
    model_path = os.path.join(model_dir, model_name + ".onnx")

    # Create the tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Create an ONNX Runtime session to run the ONNX model
    session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])  


def run(raw_data):
    logging.info("Request received")
    inputs = json.loads(raw_data)

    logging.info(inputs)

    # Preprocess the question and context into tokenized ids
    input_ids, segment_ids, tokens = preprocess(inputs["question"], inputs["context"])
  
    # Format the inputs for ONNX Runtime
    model_inputs = {
        'input_ids':   [input_ids], 
        'segment_ids': [segment_ids]
        }
                  
    outputs = session.run(['start_logits', 'end_logits'], model_inputs)

    # Format the outputs for the post processing function
    outputs_dict = {
        'start_logits': outputs[0],
        'end_logits': outputs[1]
    }

    # Post process the output of the model into an answer (or an error if the question could not be answered)
    return postprocess(tokens, outputs_dict)


if __name__ == '__main__':
    init()   
    input 
    print(run("{\"question\": \"What is a major importance of Southern California in relation to California and the United States?\", \"context\": \"Southern California, often abbreviated SoCal, is a geographic and cultural region that generally comprises California\'s southernmost 10 counties. The region is traditionally described as \'eight counties\', based on demographics and economic ties: Imperial, Los Angeles, Orange, Riverside, San Bernardino, San Diego, Santa Barbara, and Ventura. The more extensive 10-county definition, including Kern and San Luis Obispo counties, is also used based on historical political divisions. Southern California is a major economic center for the state of California and the United States.\"}"))

