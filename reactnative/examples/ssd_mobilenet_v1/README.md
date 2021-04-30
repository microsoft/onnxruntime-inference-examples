# react-native-onnxruntime-module-test

test app for react-native-onnxruntime-module

## npm package for react-native-onnxruntime-module

```sh
git clone https://github.com/hanbitmyths/react-native-onnxruntime-module
cd react-native-onnxruntime-module
yarn
npm pack
```
This wil create react-native-onnxruntime-module-0.1.0.tgz

## prerequisite

Run this from project root directory
```
yarn
npm install react-native-onnxruntime-module-0.1.0.tgz
```

## Android

This steps are already done at this example. Apply this steps only when you have new project.

Open project build.gradle and add this into allprojects/repositories
```js
flatDir {
  dir project(':reactnativeonnxruntimemodule').file('libs')
}
```
Open project settings.gradle and add this
```js
include ':reactnativeonnxruntimemodule'
project(':reactnativeonnxruntimemodule').projectDir = new File(rootProject.projectDir, '../../node_modules/react-native-onnxruntime-module/android')
```
Open project gradle/wrapper/gradle-wrapper.properties and change distributionUrl to 'https\://services.gradle.org/distributions/gradle-6.5-all.zip'
 Open module build.gradle and add this into dependencies
```js
implementation project(':reactnativeonnxruntimemodule')
```

## iOS

This steps are already done at this example. Apply this steps only when you have new project.

Add this into Podfile and run 'Pod install'
```js
pod 'react-native-onnxruntime-module', :path => '../../node_modules/react-native-onnxruntime-module'
```
For simulator, copy and rename 'libonnxruntime.1.6.0.iphonesimulator.dylib' to 'libonnxruntime.1.6.0.dylib'.
For iPhone, copy and rename 'libonnxruntime.1.6.0.iphoneos.dylib' to 'libonnxruntime.1.6.0.dylib'.

## run example

Run this commands from project root directory to execute Android or iOS example.
```
yarn bootstrap
yarn example android
yarn example ios
```

## License

MIT
