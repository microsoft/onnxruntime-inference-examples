import { StatusBar } from 'expo-status-bar';
import React, {useState} from 'react';
import { Alert, Button, StyleSheet, Text, View, NativeModules, Image } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';

import * as ort from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';
import { FlipType } from 'expo-image-manipulator';


let model: ort.InferenceSession;

//Loads in image as uint8array
let isLoaded = false;
let uint8Pixels = new Uint8Array(300 * 300 *3)
const [imgHeight, imgWidth] = [300, 300]
let bitmapPixel: number[] = Array(imgHeight*imgWidth);
const bitmapModule = NativeModules.Bitmap
let label;


export default function App() {
  const [selectedImage, setSelectedImage] = useState<any>(null);
  const [outputImage, setOutputImage] = useState<any>(null);
  const [myModel, setModel] = useState(model);

  //detects if model has successfully loaded
  async function loadModel() {
    try {
      const assets = await Asset.loadAsync(require('./assets/ssd_mobilenet_v1.opset13.exported.ort'));
      console.log(assets);
      const modelUri = assets[0].localUri;
      console.log(modelUri);
      if (!modelUri) {
        Alert.alert('failed to get model URI', `${assets[0]}`);
        console.log("failed to get model uri");
      } else {
        setModel(await ort.InferenceSession.create(modelUri));
          console.log("model loaded successfully");
      } 
    } catch (e) {
      Alert.alert('failed to load model test', `${e}`);
      console.log("failed to load model test");
      throw e;
    }
  }


  // process and inferences model
  async function runModel() {
    try {
      console.log("TEST");
      // console.log(inputData.length);
      const feeds:Record<string, ort.Tensor> = {};
      const inputData = await preprocess()
      feeds[myModel.inputNames[0]] = inputData;
      // console.log(inputData);
      const fetches = await myModel.run(feeds);
      // console.log(fetches);
      const detectionBoxes = fetches[myModel.outputNames[0]].data as Float32Array;
      const names = fetches[myModel.outputNames[1]].data as Float32Array;
      if (!detectionBoxes) {
        Alert.alert('failed to get output', `${myModel.outputNames[0]}`);
        console.log("FAILED TO GET OUTPUT");
      } else {
          console.log("MODEL INFERENCE SUCCESSFULLY")
          await postprocess(detectionBoxes, names)
          // return output
      }
    } catch (e) {
      Alert.alert('failed to inference model', `${e}`);
      console.log("FAILED TO INFERENCE MODEL")
      throw e;
    }
  }



  // opens camera roll to select an image
  async function openImagePickerAsync() {
  
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
  
    if (permissionResult.granted === false) {
      alert("Permission to access Camera Roll is Required!");
      return;
    }

    const pickerResult = await ImagePicker.launchImageLibraryAsync();
    
    if (pickerResult.cancelled === true) {
      return;
    }

    const imageResult = await ImageManipulator.manipulateAsync(
      pickerResult.uri, [
        {resize: {height: imgHeight, width: imgWidth}}
      ]
    )

    bitmapPixel = await bitmapModule.getPixels(imageResult.uri).then(
      (image: any) => {
        return Array.from(image.pixels);
      }
    )

    setSelectedImage({ 
      localUri: imageResult.uri,
      localHeight: imageResult.height,
      localWidth: imageResult.width 
    });

    setOutputImage(null)

    return
    
  };


  //Preprocessing function that uses bitmap library
  async function preprocess(): Promise<ort.Tensor> {
    // uint8Pixels.forEach((value, index) => {
    // });
    
    //Allows us to gain access to each pixel so we can run the model properly
    bitmapPixel.forEach((value, index) => {
      const pixel = value
      const postIndex = index * 3
      let red = (pixel >> 16 & 0xFF)
      let green = (pixel >> 8 & 0xFF)
      let blue = (pixel & 0xFF)

      // red = (((red) - 0.485) / 0.229)
      // green = (((green) - 0.456) / 0.224)
      // blue = (((blue) - 0.406) / 0.225)
      
      uint8Pixels[postIndex] = red
      uint8Pixels[postIndex + 1] = green
      uint8Pixels[postIndex + 2] = blue
    })
    let tensor: ort.Tensor = new ort.Tensor(uint8Pixels, [1, imgHeight, imgWidth, 3])
    return tensor
  };

  //object detection bounding box post processing output
  async function postprocess(det_boxes: Float32Array, names: Float32Array) {
    let labels =  new Map();
    for(let i = 0; i < names.length; i++){
      const name = names[i]
      const det_index = i * 4
      //Outputs first 3 labels detected
      if (labels.size >= 1){
        break
      } else if (labels.has(name)){
        continue
      } else{
        labels.set(name, Array.of(det_boxes[det_index], det_boxes[det_index+1], det_boxes[det_index+2], det_boxes[det_index+3]))
      }
    }
    console.log(labels)
    //Iterates through the boxes and calls detProcess function
    for (const [name, detBox] of labels) {
      await detProcess(detBox)
    }
    console.log("Done")
    //Process that converts the pixels to red
    let intPixels = Array<number>(300*300)
    for (let i = 0; i < uint8Pixels.length; i += 3){
      const red = uint8Pixels[i]
      const green = uint8Pixels[i+1]
      const blue = uint8Pixels[i+2]

      intPixels[Math.floor(i/3)] = ((0xFF << 24) |
                        ((0xFF & blue) << 16) |
                        ((0xFF & green) << 8) |
                        ((0xFF & red)))
    }
    let imageUri = await bitmapModule.getImageUri(intPixels).then(
      (image:any) => {

        return image.uri
      }
    )
    const imageRotated = await ImageManipulator.manipulateAsync(imageUri, [
      {rotate: 90},
      {flip: ImageManipulator.FlipType.Horizontal}
    ])
    setOutputImage({ localUri: imageRotated.uri })
    

  }


  //Draws all four lines for bounding box, 2 horizontal and 2
  async function detProcess(array: number[]) {
    array.forEach((value, index) => {
      array[index] = Math.floor(Math.max(Math.min(value * 300, 300), 0))
    })
    const topLeft = Array.of(array[0], array[1])
    const topRight = Array.of(array[0], array[3])
    const bottomLeft = Array.of(array[2], array[1])
    const bottomRight = Array.of(array[2], array[3])
    
    await drawPixel(topLeft, topRight, "H")
    await drawPixel(bottomLeft, bottomRight, "H")
    await drawPixel(topLeft, bottomLeft, "V")
    await drawPixel(topRight, bottomRight, "V")

  }

  //function that gains access to each individual pixel and converts them to red
  async function drawPixel(first:number[], last: number[], direction: string) {
    //For drawing horizontal lines of box
    if (direction == "H"){
      const row = first[0]
      for (let col = first[1]; col < last[1]; col += 3){
        for (let i = row-1; i <= row+1; i++){
          if (i > 300 || i < 0){
            continue
          }
          for (let j = col-1; j <= col+1; j++){
            if (j > 300 || j < 0){
              continue
            }
            const redPos = ((i-1)*300 + (j-1)) * 3
            const greenPos = redPos + 1
            const bluePos = redPos + 2
            uint8Pixels[redPos] = 255
            uint8Pixels[greenPos] = 0
            uint8Pixels[bluePos] = 0
          }
        }
      }
    }else{
      //draws red line for vertical lines
      const col = first[1]
      for (let row = first[0]; row < last[0]; row += 3){
        for (let i = row-1; i <= row+1; i++){
          if (i > 300 || i < 0){
            continue
          }
          for (let j = col-1; j <= col+1; j++){
            if (j > 300 || j < 0){
              continue
            }
            const redPos = ((i-1)*300 + (j-1)) * 3
            const greenPos = redPos + 1
            const bluePos = redPos + 2
            uint8Pixels[redPos] = 255
            uint8Pixels[greenPos] = 0
            uint8Pixels[bluePos] = 0
          }
        }
      }

    }
    
  }


  return (
    <View style={styles.container}>
      <Text>using ONNX Runtime for React Native</Text>
      <Button title='Load model' onPress={loadModel}></Button>
      <Button title='Run' onPress={runModel}></Button>
      <Button title="Upload image" onPress={openImagePickerAsync}></Button>
      {selectedImage &&
      <Image
        source={{ uri: selectedImage.localUri}}
        style={styles.thumbnail}
      ></Image>}
      {outputImage &&
      <View style={{alignItems: "center", justifyContent: "center"}}>
      <Image
        source={{ uri: outputImage.localUri}}
        style={styles.thumbnail}
      ></Image>

      
      </View>}

      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  thumbnail: {
    alignSelf: "center",
    width: 300,
    height: 300,
    resizeMode: "contain"
  },
});
