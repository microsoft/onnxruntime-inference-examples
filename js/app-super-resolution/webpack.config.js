const createExpoWebpackConfigAsync = require('@expo/webpack-config');
const CopyPlugin = require("copy-webpack-plugin");

module.exports = async function (env, argv) {
  const config = await createExpoWebpackConfigAsync(env, argv);
  // Customize the config before returning it.

  config.plugins.push(
    new CopyPlugin({
      patterns: [
        {
          from: './node_modules/onnxruntime-web/dist/ort-wasm.wasm',
          to: 'static/js'
        },             {
          from: './node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm',
          to: 'static/js'
        },
        {
          from: './node_modules/onnxruntime-web/dist/ort-wasm-threaded.wasm',
          to: 'static/js'
        },             {
          from: './node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm',
          to: 'static/js'
        }]          
      }
    )
  )


  return config;
};
