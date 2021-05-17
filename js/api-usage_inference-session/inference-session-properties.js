// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const ort = require('onnxruntime-node');

// following code also works for onnxruntime-web.

const InferenceSession = ort.InferenceSession;

// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create session and load model.onnx
        const session = await InferenceSession.create('./model.onnx');;

        //
        // get input/output names from inference session object
        //
        const inputNames = session.inputNames;
        const outputNames = session.outputNames;

    } catch (e) {
        console.error(`failed to create inference session: ${e}`);
    }
}

main();
