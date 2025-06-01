# Object Detection Implementation for Android Mobile using ONNX Runtime

Steps of building this project from scratch

0. Prepare environment

     i. Install Node.js

     ii. Install [expo](https://docs.expo.dev/)
     ```
     npm install -g expo-cli
     ```
     iii. Install yarn
     ```
     npm install -g yarn
     ```
     iv. Install onnxruntime-react-native
     ```
     expo install onnxruntime-react-native@dev
     ```
  1. Project Setup

      i. Clone this repo 
      
      ii. Navifate to the ```js\ExpoObjectDetection``` folder. This will serve as your ```<SOURCE_ROOT>```
  2. Add the object detection model to the project.

       i. Download the ORT model found [here](https://github.com/vasquezd21/ExpoObjectDetection/blob/master/assets/ssd_mobilenet_v1.opset13.exported.ort) and add this model into the ```assets``` folder in the ```<SOURCE_ROOT>```

  3. Install required libraries

       i. There are two methods to install the required libraries. Run either of the following commands below in ```<SOURCE_ROOT>``` to install the modules and libraries specified in the package.json file.

       - Using YARN:
       ```
       yarn
       ```
       - Using NPM:
       ```
       npm install
       ```
       **NOTE:** if you run into dependency issues and the installation fails, you can run this instead to install the modules.
       ```
       yarn --force
       ```
  4. Setup Android Projects:

       i. In ```<SOURCE_ROOT>```, run the following command to generate project files.
       ```
       expo prebuild
       ```
       ii. Move the Move the ```BitmapModule.java``` file and the ```BitmapReactPackage.java``` file in ```<SOURCE_ROOT>``` to this directory: ```<SOURCE_ROOT>\android\app\src\main\java\com\dmvasquez\ExpoObjectDetection``` These files contain native module functions useful for the Android implementation. For more information on Native Modules, click this link: [Native Modules (React Native)](https://reactnative.dev/docs/next/native-modules-intro)

       iii. In ```<SOURCE_ROOT>\android\app\src\main\java\com\dmvasquez\oMainApplication.java```, add the following lines of code to the ```getPackages``` function:
       ```
       @Override
      protected List<ReactPackage> getPackages() {
      @SuppressWarnings("UnnecessaryLocalVariable")
      List<ReactPackage> packages = new PackageList(this).getPackages();
      // Packages that cannot be autolinked yet can be added manually here, for example:
       // packages.add(new MyReactNativePackage());
      packages.add(new BitmapReactPackage());
      return packages;
      }
       ```
  5. Run the following command to launch project,
  - In ```<SOURCE_ROOT>```, run the following to launch on Android
    ```
    expo run:android
    ```

