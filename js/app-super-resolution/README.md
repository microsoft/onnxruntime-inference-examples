# RUNNING SUPER-RESOLUTION ON IMAGES ACROSS VARIOUS PLATFORMS USING ONNX RUNTIME MODELS
## Examples
1. MOBILE
  <img width="318" alt="image" src="https://user-images.githubusercontent.com/106185642/181639530-9c808167-d68c-49d4-8e89-72aeeb11164e.png">

2. WEB
  <img width="1215" alt="image" src="https://user-images.githubusercontent.com/106185642/181638855-f341e52e-dfc1-4362-b93a-0117f0cfd65a.png">
   

## Steps of building project from scratch
0. Prerequisites
    1. Install Node.js
    2. Install Java and OpenJDK
    3. Install expo
       ```sh
       npm install -g expo-cli
       ```
    4. Install yarn
       ```sh
       npm install -g yarn
       ```
       
1. Setup project
    1. Create Root Directory
    2. Copy contents of this Repo into created directory

    **NOTE:**
    - `<SOURCE_ROOT>` refers to the root folder of the source code directory.
 
2. Install required Libraries
   1. Run the following command in `<SOURCE_ROOT>` to install the modules and libraries specified in the package.json file.
      1. Using YARN:
         ```sh
         yarn
         ```
      
      2. Using NPM:
         ```sh
         npm install
         ```

         
     **NOTE:**
     If you run into dependency issues, you can run this instead to install the modules.
      ```sh
      npm install --force
      ```
   
3. Run the following commands to launch:

    In `<SOURCE_ROOT>`, run the following command to launch for Android
    ```sh
    expo run:android
    ```

    In `<SOURCE_ROOT>`, run the following command to launch for Web
    ```sh
    expo start --web
    ```
