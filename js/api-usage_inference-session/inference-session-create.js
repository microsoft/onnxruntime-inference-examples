// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const fs = require('fs');
const util = require('util');
const ort = require('onnxruntime-node');

// following code also works for onnxruntime-web.

const InferenceSession = ort.InferenceSession;

// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create session option object
        const options = createMySessionOptions();

        //
        // create inference session from a ONNX model file path or URL
        //
        const session01 = await InferenceSession.create('./model.onnx');
        const session01_B = await InferenceSession.create('./model.onnx', options); // specify options

        //
        // create inference session from an Node.js Buffer (Uint8Array)
        //
        const buffer02 = await readMyModelDataFile('./model.onnx'); // buffer is Uint8Array
        const session02 = await InferenceSession.create(buffer02);
        const session02_B = await InferenceSession.create(buffer02, options); // specify options

        //
        // create inference session from an ArrayBuffer
        //
        const arrayBuffer03 = buffer02.buffer;
        const offset03 = buffer02.byteOffset;
        const length03 = buffer02.byteLength;
        const session03 = await InferenceSession.create(arrayBuffer03, offset03, length03);
        const session03_B = await InferenceSession.create(arrayBuffer03, offset03, length03, options); // specify options

        // example for browser
        //const arrayBuffer03_C = await fetchMyModel('./model.onnx');
        //const session03_C = await InferenceSession.create(arrayBuffer03_C);
    } catch (e) {
        console.error(`failed to create inference session: ${e}`);
    }
}

main();

function createMySessionOptions() {
    // session options: please refer to the other example for details usage for session options

    // example of a session option object in node.js:
    // specify intra operator threads number to 1 and disable CPU memory arena
    return { intraOpNumThreads: 1, enableCpuMemArena: false }

    // example of a session option object in browser:
    // specify WebAssembly exection provider
    //return { executionProviders: ['wasm'] };

}

async function readMyModelDataFile(filepathOrUri) {
    // read model file content (Node.js) as Buffer (Uint8Array)
    return await util.promisify(fs.readFile)(filepathOrUri);
}

async function fetchMyModel(filepathOrUri) {
    // use fetch to read model file (browser) as ArrayBuffer
    if (typeof fetch !== 'undefined') {
        const response = await fetch(filepathOrUri);
        return await response.arrayBuffer();
    }
}