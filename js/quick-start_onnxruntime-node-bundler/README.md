# Quick Start - Node (using bundler)

This example is a demonstration of basic usage of ONNX Runtime Node, using a bundler.

A bundler is a tool that puts your code and all its dependencies together in one JavaScript file. In this example, we use [webpack](https://webpack.js.org) to pack our code, and comsume the generated bundle.js in our HTML.

When using webpack on nodejs modules it is not common to include modules like onnxruntime in the bundle since that results in bloat of the bundle and defeats the purpose of the npm package management. For that reason this example uses [webpack-node-externals](https://www.npmjs.com/package/webpack-node-externals) to exclude it from the package.

## Usage

1. install dependencies:
   ```sh
   npm install
   ```

2. use webpack to make bundle:
   ```sh
   npm run build
   ```
   this generates the bundle file `./dist/main.js`

3. run with
   ```sh
   node dist/main.js
   ```
   