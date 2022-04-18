import os
import logging
import json
import numpy as np
import onnxruntime
import transformers
import torch


def preprocess(question, context):
    print("Question:", question)
    print("Context: ", context)
    encoded_input = tokenizer(question, context)
    tokens = tokenizer.convert_ids_to_tokens(encoded_input.input_ids)
    print(tokens)
    return (encoded_input.input_ids, encoded_input.attention_mask, encoded_input.token_type_ids, tokens)

def postprocess(tokens, start, end):
    print("Start:", start)
    print("End:", end)
    results = {}
    answer_start = np.argmax(start)
    answer_end = np.argmax(end)
    print("Start: ", answer_start)
    print("End: ", answer_end)
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


def run_pytorch(raw_data):
    inputs = json.loads(raw_data)

    model.eval()

    logging.info("Question:", inputs["question"])
    logging.info("Context: ", inputs["context"])

    input_ids, input_mask, segment_ids, tokens = preprocess(inputs["question"], inputs["context"])

    print(input_ids)

    model_outputs = model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([segment_ids]))

    outputs = postprocess(tokens, model_outputs.start_logits.detach().numpy(), model_outputs.end_logits.detach().numpy())

    print(outputs)


def run(raw_data):
    logging.info("Request received")
    inputs = json.loads(raw_data)

    logging.info(inputs)

    # Preprocess the question and context into tokenized ids
    input_ids, input_mask, segment_ids, tokens = preprocess(inputs["question"], inputs["context"])

    print(input_ids)
    
    # Format the inputs for ONNX Runtime
    model_inputs = {
        'input_ids':   [input_ids], 
        'input_mask':  [input_mask],
        'segment_ids': [segment_ids]
        }
                  
    outputs = session.run(['start_logits', 'end_logits'], model_inputs)
    
    # Post process the output of the model into an answer (or an error if the question could not be answered)
    return postprocess(tokens, outputs[0], outputs[1])


if __name__ == '__main__':
    init()

    #input = "{\"question\": \"What is my name?\", \"context\": \"My name is Natalie, and my sister's name is also Nathalie and my brother's name is dufas and my friend's name is Remy\"}"

    #input = "{\"question\": \"What is Dolly Parton's middle name?\", \"context\": \"Dolly Rebecca Parton (born January 19, 1946) is an American singer-songwriter, actress, and businesswoman, known primarily for her work in country music. After achieving success as a songwriter for others, Parton made her album debut in 1967 with Hello, I'm Dolly, which led to success during the remainder of the 1960s (both as a solo artist and with a series of duet albums with Porter Wagoner), before her sales and chart peak came during the 1970s and continued into the 1980s. Parton's albums in the 1990s did not sell as well, but she achieved commercial success again in the new millennium and has released albums on various independent labels since 2000, including her own label, Dolly Records. She has sold more than 100 million records worldwide.\"}"

    input = "{\"question\": \"What is Dolly Parton's middle name?\", \"context\": \"Dolly Rebecca Parton is an American singer-songwriter\"}"

    run_pytorch(input)
    print(run(input))

