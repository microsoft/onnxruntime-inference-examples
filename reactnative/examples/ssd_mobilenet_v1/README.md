# onnxrutnime-react-native-test

test app for onnxrutnime-react-native

## prerequisite

Run this from project root directory
```
yarn add onnxruntime-react-native
```

## Android

This steps are already done at this example. Apply this steps only when you have new project.

Open project build.gradle and add this into allprojects/repositories
```js
flatDir {
  dir project(':onnxruntimereactnative').file('libs')
}
```
Open project settings.gradle and add this
```js
include ':onnxruntimereactnative'
project(':onnxruntimereactnative').projectDir = new File(rootProject.projectDir, '../../node_modules/onnxrutnime-react-native/android')
```
Open project gradle/wrapper/gradle-wrapper.properties and change distributionUrl to 'https\://services.gradle.org/distributions/gradle-6.5-all.zip'
 Open module build.gradle and add this into dependencies
```js
implementation project(':onnxruntimereactnative')
```

## iOS

This steps are already done at this example. Apply this steps only when you have new project.

Add this into Podfile and run 'Pod install'
```js
pod 'onnxrutnime-react-native.iphonesimulator', :path => '../../node_modules/onnxrutnime-react-native'
```

If your taget is iOS, use `onnxruntime-react-native.iphoneos` instead of `onnxruntime-react-native.iphonesimulator`.

## run example

Run this commands from project root directory to execute Android or iOS example.
```
yarn
yarn bootstrap
yarn example android
yarn example ios
```

## License

License information can be found [here](https://github.com/microsoft/onnxruntime-inference-examples/blob/master/README.md#license).
