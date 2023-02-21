// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const ort = require('onnxruntime-node');

// following code also works for onnxruntime-web.

const Tensor = ort.Tensor;

const myTensors = getMyTensors();

//
// get dimensions and size from tensor
//
const dims_0 = myTensors[0].dims; // [2, 3, 4]
const size_0 = myTensors[0].size; // 24

//
// get type from tensor
//
const type_1 = myTensors[1].type; // 'bool'

//
// get data (typed array) from tensor
//
const data_2 = myTensors[2].data; // Int32Array(6)


function getMyTensors() {
    const buffer01 = new Float32Array(24).fill(1);

    return [
        new Tensor('float32', buffer01, [2, 3, 4]),
        new Tensor('bool', [true], []),
        new Tensor('int32', [1, 2, 3, 4, 5, 6], [2, 3])
    ];
}