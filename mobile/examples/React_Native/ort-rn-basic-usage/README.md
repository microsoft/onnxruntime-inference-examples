# ONNX Runtime React Native Expo Example Application

This is a [React Native](https://reactnative.dev/docs/getting-started) application for [ONNX Runtime](https://github.com/microsoft/onnxruntime) via [Expo](https://docs.expo.dev/) platform. The demo app demonstrates how to accomplish simple tasks such as loading onnx models and creating inference sessions, etc. in a react native expo application.

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


## Validate your React Native Environment

Run the example expo app as-is for validating local React Native environment setup.

**Steps:**

1. Run `yarn install` to set up JavaScript dependencies.
    ```sh
    yarn install
    ```

2. Install NPM `onnxruntime-react-native` package.
    ```sh
    expo install onnxruntime-react-native@latest
    ```
 
3. Run the following command in `<PROJECT_ROOT>` to generate Android and iOS project.
        ```sh
        expo prebuild
        ```
    The generated Android and iOS project code will be updated automatically.

4. Build and run the app. 

    Run the following command to build and launch the app:

    - In `<PROJECT_ROOT>`, run the following command to launch for Android:
        
    ```sh
        expo run:android
    ```

    - In `<PROJECT_ROOT>`, run the following command to launch for iOS:
    ```sh
        expo run:ios
    ```

## Steps that were done to add onnxruntime-react-native to the example app

The following steps were done in this sample for using onxnruntime-react-native. These can be replicated as a reference when setting up your own react native expo application.

1. NPM `onnxruntime-react-native` package. 

   Note: By default, we install the current latest release version of ORT react native npm package(Recommended). Can also update to dev version npm package if necessary.

   [Link to npm `onnxruntime-react-native` packages](https://www.npmjs.com/package/onnxruntime-react-native?activeTab=versions)

2. Prepare the model.

    -  Model files are usually placed under `<PROJECT_ROOT>/assets`.
    
       In this sample application, a simple ONNX MNIST model (`mnist.onnx`) is provided by default.

    -  Add file `metro.config.js` under `<PROJECT_ROOT>`. This file adds the extension `.onnx` to the bundler's asset extension list, which allows the bundler to include the model into assets.
       
       `metro.config.js` file in this sample application looks like:

       ```js
       const { getDefaultConfig } = require('@expo/metro-config');
       const defaultConfig = getDefaultConfig(__dirname);
       defaultConfig.resolver.assetExts.push('onnx');
       module.exports = defaultConfig;
       ```

3. Generate Android and iOS directories native code to run your React Native app.
    
    In this sample project, it generates the native code automatically by using package `onnxruntime-react-native` as an Expo plugin. (Recommended approach.)
    
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
**NOTE:**
   If you are interested in creating a new react native expo project from scratch, refer to instructions: https://docs.expo.dev/get-started/create-a-project/