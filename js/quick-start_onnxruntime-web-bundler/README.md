# Quick Start - Web (using bundler)

This example is a demonstration of basic usage of ONNX Runtime Web, using a bundler.

A bundler is a tool that puts your code and all its dependencies together in one JavaScript file. In this example, we use [webpack](https://webpack.js.org) to pack our code, and comsume the generated bundle.js in our HTML. See also [Quick Start - Web (using script tag)](../quick-start_onnxruntime-web-script-tag) for an example of using script tag.

Modern browser based applications are usually built by frameworks like [Angular](https://angularjs.org/), [React](https://reactjs.org/), [Vue.js](https://vuejs.org) and so on. Those frameworks usually utilize bundler plugins to build the whole application. To keep our example simple and small, we will not use those frameworks, but the usage of bundler should be similar.

This example contians a `package.json` file, which already lists "onnxruntime-web" as dependency and webpack packages as development dependency. To work on your own `package.json`, use command `npm install onnxruntime-web` to install ONNX Runtime Web, and use command `npm install --save-dev webpack webpack-cli` to install webpack as development dependency.

[Webpack](https://webpack.js.org) is a very powerful bundler. This example uses a simple config file `webpack.config.js` to shows basic usage only. See also Webpack's [Getting Started](https://webpack.js.org/guides/getting-started/) for more information.

In this example, we load onnxruntime, create an inference session with a simple model, feed input, get output as result and write it to the HTML page. All functions are called in their basic form.

## Usage

1. install dependencies:
   ```sh
   npm install
   ```

2. use webpack to make bundle:
   ```sh
   npx webpack
   ```
   this generates the bundle file `./dist/bundle.min.js`

3. use NPM package `light-server` to serve the current folder at http://localhost:8080/
   ```sh
   npx light-server -s . -p 8080
   ```

4. open your browser and navigate to the URL.