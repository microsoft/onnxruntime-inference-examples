## Steps of building project from scratch (For Windows)

0. Prerequisites

   1. Install Node.js

   2. Install Java (OpenJDK).

      - Download and install [OpenJDK Version 11](https://adoptopenjdk.net/).
      - Set JAVA_HOME environment variable. Steps are highlighted [here](https://java2blog.com/how-to-set-java-path-windows-10/#How_to_set_JAVA_HOME_in_Windows_10).

   3. Install the [expo-cli](https://docs.expo.dev/)

      ```sh
      npm install -g expo-cli
      ```

   4. Install yarn
      ```sh
      npm install -g yarn
      ```

1. Setup project

   1. Clone this Repo

   2. Navigate to the `js\app-super-resolution` folder. This folder will be your `<SOURCE_ROOT>`

2. Add ORT model to project:

   - Download the ORT model from this [link](https://github.com/VictorIyke/super_resolution_MW/blob/main/cross_plat/assets/super_resnet12.ort) into the `assets` folder in the `<SOURCE_ROOT>`

3. Install required Libraries

   1. There are two ways to do this. Run any of the commands below in `<SOURCE_ROOT>` to install the modules and libraries specified in the package.json file.

      - Using YARN (Recommended):

        ```sh
        yarn
        ```

      - Using NPM:

        ```sh
        npm install
        ```

      **NOTE:**
      If you run into dependency issues and the installation fails, you can run this instead to install the modules.

      ```sh
      yarn --force
      ```

4. Setup Android Projects:

   1. In `<SOURCE_ROOT>`, run the following command to generate android project files.

      ```sh
      expo prebuild
      ```

      The generated project files will be updated automatically.

   2. Move the BitmapModule.java file and the BitmapReactPackage.java file in `<SOURCE_ROOT>` to this directory:
      `<SOURCE_ROOT>\android\app\src\main\java\com\example\ortdemo`
      These files contain native module functions useful for the Android implementation.
      For more information on Native Modules, click this link: [Native Modules (React Native)](https://reactnative.dev/docs/next/native-modules-intro)

   3. In `<SOURCE_ROOT>\android\app\src\main\java\com\example\ortdemo\MainApplication.java`, add the following lines of code to the `getPackages` function:

      ```java
      packages.add(new BitmapReactPackage());
      ```

      The resulting function will look like this:

      ```java
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

5. Run the following command to launch:

   In `<SOURCE_ROOT>`, run the following command to launch for Android

   ```sh
   expo run:android
   ```

   In `<SOURCE_ROOT>`, run the following command to launch for Web

   ```sh
   expo start --web
   ```

* Note:
  * To try the updated data pre/post processing support for android side, currently need to install 
  * 1. onnxruntime-react-native custom built package with ort-extensions dependency.
  * 2. `base64-js` and `react-native-quick-base64` npm package for base64 encoding/decoding support.
  * TODO: update above required dependencies.
