# ONNX Runtime React Native Sample Usage Application with ort-extensions

## Overview

This is a basic example usage react native expo application for [ONNX Runtime](https://github.com/microsoft/onnxruntime) with [Ort-Extensions](https://github.com/microsoft/onnxruntime-extensions) support for pre/post processing. The demo app accomplishes the task of loading a model with pre/post processing ops and run inference session to get the output result.

0. Prepare environment
    1. install Node.js
    2. install expo

        ```sh
        npm install -g expo-cli
        ```

    3. install yarn

        ```sh
        npm install -g yarn
        ```

1. Setup empty project

   ```sh
   cd <SOURCE_ROOT>
   expo init . -t expo-template-blank-typescript
   yarn
   ```

   **NOTE:**
   - `<SOURCE_ROOT>` refers to the root folder of the source code, where this `README.md` file sits.
   i.e. `/mobile/examples/ort_extensions_react_native_usage/ort-ex-rn-usage`

2. Install onnxruntime-react-native

    ```sh
    expo install onnxruntime-react-native@dev
    ```
3. Add your ONNX model to project

    1. Put the file under `<SOURCE_ROOT>/assets`.

       In this tutorial, we use test sample ONNX model with custom op decode_image (`decode_image.ort`).

    2. add a new file `metro.config.js` under `<SOURCE_ROOT>` and add the following lines to the file:

       ```js
       const { getDefaultConfig } = require('@expo/metro-config');
       const defaultConfig = getDefaultConfig(__dirname);
       defaultConfig.resolver.assetExts.push('onnx');
       module.exports = defaultConfig;
       ```

       This step adds extension `onnx` to the bundler's asset extension list, which allows the bundler to include the model into assets.

4. Setup Android and iOS project.

    We use expo prebuild steps to generate Android/iOS projects folder to consume ONNX Runtime and Ort-Extensions.

    - Use NPM package `onnxruntime-react-native` as an expo plugin.
        1. In `<SOURCE_ROOT>/app.json`, add the following line to section `expo`:

           ```
           "plugins": ["onnxruntime-react-native"],
           ```

        2. Run the following command in `<SOURCE_ROOT>` to generate Android and iOS project:

            ```sh
            expo prebuild
            ```

        The generated project files will be updated automatically.

5. Enable Ort Extensions in React Native app.
   1. In `<SOURCE_ROOT>/package.json` file, specify the field to build expo project with ort-extensions package:

        ```
        "ortPackageName": "onnxruntime-ext"
        ```

   2. Set up Android project to register custom ops for ort-extensions.

      In `<SOURCE_ROOT>/node_modules/onnxruntime-react-native/android/src/main/java/ai/onnxruntime/reactnative/OnnxruntimeModule.java`,

      Uncomment the following line:

      ```diff
      // Optional call to register custom ops
      - // sessionOptions.registerCustomOpLibrary(ai.onnxruntime.extensions.OrtxPackage.getLibraryPath());
      + sessionOptions.registerCustomOpLibrary(ai.onnxruntime.extensions.OrtxPackage.getLibraryPath());
      ```

   3. Set up iOS project to register custom ops for ort-extensions.

      - In `<SOURCE_ROOT>/node_modules/onnxruntime-react-native/ios/OnnxruntimeModule.mm`,

        Uncomment the following lines:

        ```diff
        // Optional include of ort extensions header
        - // #include <onnxruntime_extensions.h>
        + #include <onnxruntime_extensions.h>
        ```

        ```diff
        /*
           Optional call of to enable usage of ort extensions custom ops
        */
        - // if (RegisterCustomOps(sessionOptions, OrtGetApiBase()) != nullptr) {
        - // throw std::runtime_error("RegisterCustomOps failed");
        - // }
        + if (RegisterCustomOps(sessionOptions, OrtGetApiBase()) != nullptr) {
        +    throw std::runtime_error("RegisterCustomOps failed");
        + }
        ```

6. Add code in `App.tsx` to use onnxruntime-react-native with extensions.

    Please refer to the file content for more details.

    Two main basic usage functions:
    ```
    - async function loadModel()
    - async function runModel()
    ```

7. Run the following command to launch:

    In `<SOURCE_ROOT>`, run the following command to launch for Android

    ```sh
    expo run:android
    ```

    In `<SOURCE_ROOT>`, run the following command to launch for iOS

    ```sh
    expo run:ios
    ```
    Note: for iOS project, need to apply the following manual step currently:
    
    In `<SOURCE_ROOT>/ios/Pods/onnxruntime-extensions-c/Headers/onnxruntime_extensions.h`,
    
    ```diff
    -  #include "onnxruntime_cxx_api.h"
    +  #include <onnxruntime/onnxruntime_cxx_api.h>
    ```
