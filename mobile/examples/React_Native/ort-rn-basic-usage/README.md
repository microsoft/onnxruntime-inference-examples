# ONNX Runtime React Native Expo Example Application

This is a [react native](https://reactnative.dev/docs/getting-started) application for [ONNX Runtime](https://github.com/microsoft/onnxruntime) via [Expo](https://docs.expo.dev/) platform. The demo app demonstrates how to accomplish simple tasks such as loading onnx models and creating inference sessions, etc. in a react native expo application.

## Prerequisites

1. Install [Node.js](https://nodejs.org/en)
2. Install [Expo-CLI](https://docs.expo.dev/more/expo-cli/)
    ```sh
    npm install -g expo-cli
    ```
3. Install [Yarn](https://classic.yarnpkg.com/en/docs/install#mac-stable) (Recommended)
    ```sh
    npm install -g yarn
    ```
**NOTE:**
   For creating a new react native expo project from scratch, refer to: https://docs.expo.dev/get-started/create-a-project/

## Set up

1. Install NPM `onnxruntime-react-native` package.
    ```sh
    expo install onnxruntime-react-native@dev
    ```

2. Prepare the model.

    -  Model files are usually placed under `<PROJECT_ROOT>/assets`.
    
       In this sample application, a simple ORT format MNIST model (`mnist.ort`) is provided.

    -  File `metro.config.js` under `<PROJECT_ROOT>` adds the extension `ort` to the bundler's asset extension list, which allows the bundler to include the model into assets.
       
       `metro.config.js` file in this sample application looks like:

       ```js
       const { getDefaultConfig } = require('@expo/metro-config');
       const defaultConfig = getDefaultConfig(__dirname);
       defaultConfig.resolver.assetExts.push('ort');
       module.exports = defaultConfig;
       ```
       Adjust the extension to `.onnx` accordingly if you are loading a ONNX model in your application.

3. Generate Android and iOS directories native code to run your React app.
    
    In this sample project, it's recommended to set up the native code automatically by using package `onnxruntime-react-native` as an Expo plugin.
    
    - In `<PROJECT_ROOT>/app.json`, add the following line to section `expo`:
        ```
        "plugins": ["onnxruntime-react-native"],
        ```
        Note: The plugin is added by default in `app.json` in the sample.

    - Run the following command in `<PROJECT_ROOT>` to generate Android and iOS project: More info about [Expo CLI Prebuild](https://docs.expo.dev/workflow/prebuild/).
        ```sh
        expo prebuild
        ```
        The generated Android and iOS project code will be updated automatically.


    [Optional] Set up manually.

    1. First run  `expo prebuild` to generate Android and iOS Native code.

        Note: In this tutorial we use `ai.onnxruntime.example.reactnative.basicusage` as Android package name/iOS bundle identifier.
        Expo requires a manual configuration on package name and bundle identifier.

    2. [Android]: Add `onnxruntime-react-native` as Gradle dependencies.

        In `<PROJECT_ROOT>/android/app/build.gradle`, add the following line to section `dependencies`:

        ```
        implementation project(':onnxruntime-react-native')
        ```

    3. [iOS]: Add `onnxruntime-react-native` as CocoaPods dependencies.

        In `<PROJECT_ROOT>/ios/Podfile`, add the following line to section `target 'OrtReactNativeBasicUsage'`:

        ```sh
        pod 'onnxruntime-react-native', :path => '../node_modules/onnxruntime-react-native'
        ```

        Run the following command in `<PROJECT_ROOT>/ios` to install pod:

        ```sh
        pod install
        ```
## Build and run

Run the following command to build and launch the app:

- In `<PROJECT_ROOT>`, run the following command to launch for Android:
    ```sh
    expo run:android
    ```

- In `<PROJECT_ROOT>`, run the following command to launch for iOS:
    ```sh
    expo run:ios
    ```

