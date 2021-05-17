// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const ort = require('onnxruntime-node');

// following code also works for onnxruntime-web.

const InferenceSession = ort.InferenceSession;
const Tensor = ort.Tensor;

// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create session and load model.onnx
        const session = await InferenceSession.create('./model.onnx');

        // prepare inputs
        const dataA = prepareDataA(); // Float32Array(12)
        const dataB = prepareDataB(); // Float32Array(12)
        const tensorA = new ort.Tensor('float32', dataA, [3, 4]);
        const tensorB = new ort.Tensor('float32', dataB, [4, 3]);

        // prepare feeds. use model input names as keys.
        const feeds = {
            a: new Tensor('float32', dataA, [3, 4]),
            b: new Tensor('float32', dataB, [4, 3])
        };

        // run options
        const option = createRunOptions();

        //
        // feed inputs and run
        //
        const results_02 = await session.run(feeds);
        const results_02_B = await session.run(feeds, option); // specify options

        //
        // run with specified names of fetches (outputs)
        //
        const results_03 = await session.run(feeds, ['c']);
        const results_03_B = await session.run(feeds, ['c'], option); // specify options

        //
        // run with fetches (outputs) as nullable map
        //
        const results_04 = await session.run(feeds, { c: null });
        const results_04_B = await session.run(feeds, { c: null }, option); // specify options

        //
        // run with fetches (outputs) as nullable map, with tensor as value
        //
        const preAllocatedTensorC = new Tensor(new Float32Array(9), [3, 3]);
        const results_05 = await session.run(feeds, { c: preAllocatedTensorC });
        const results_05_B = await session.run(feeds, { c: preAllocatedTensorC }, option); // specify options

    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

main();

function prepareDataA() {
    return Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
}
function prepareDataB() {
    return Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
}

function createRunOptions() {
    // run options: please refer to the other example for details usage for run options

    // specify log verbose to this inference run
    return { logSeverityLevel: 0 };
}