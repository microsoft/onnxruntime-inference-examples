// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");
const nodeExternals = require('webpack-node-externals');

module.exports = {
    entry: {
        main: './main.js',
    },
    target: 'node',
    node: {
        __dirname: false,
    },
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'bundle.min.js',
    },
    module: {
        rules: [
        ]
    },
    plugins: [
        new CopyPlugin({
            patterns: [
                { from: 'model.onnx', to: 'model.onnx' },
            ]
        })
    ],
    externals: [
        nodeExternals(),
    ],
}