import { Button, Text, TouchableOpacity, View, Image } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { styles } from '../misc/styles';
import { Asset } from 'expo-asset';
import React, {useState} from 'react';
import { Platform } from 'expo-modules-core';
import * as ImageManipulator from 'expo-image-manipulator';
import { MainScreenProps } from './NavigStack';
import { pixelsRGBToYCbCr, pixelsYCbCrToRGB } from '../misc/utilities';


const platform = Platform.OS

let model: any;
let ortWeb: any;

if (platform == "web") {ortWeb = require("onnxruntime-web")}

let isLoaded = false;
const imageDim = 224
const scaledImageDim = imageDim * 3
let floatPixelsY = new Float32Array(imageDim * imageDim)
let cbArray = new Float32Array(scaledImageDim * scaledImageDim)
let crArray = new Float32Array(scaledImageDim * scaledImageDim)
let bitmapPixel: number[] = Array(imageDim*imageDim);
let bitmapScaledPixel: number[] = Array(scaledImageDim*scaledImageDim);

let myImageScaledData: ImageData


let offscreen: any 
if (platform == "web") {offscreen = new OffscreenCanvas(1000, 1000)}
let kdv: OffscreenCanvasRenderingContext2D | null



export default function WebApp({navigation, route}: MainScreenProps) {
  
  const [selectedImage, setSelectedImage] = useState<any>(null);
  const [myModel, setModel] = useState(model);


  if (!isLoaded || !myModel) {

    loadModel().then(() => {
      isLoaded = true;
    })
  } 

  async function loadModel() {
    try {
      const assets = await Asset.loadAsync(require('../assets/super_resnet12.ort'));
      const modelUri = assets[0].localUri;
  
      if (!modelUri) {
        console.log("Model loaded unsuccessfully")
      } else {
        setModel(await ortWeb.InferenceSession.create(modelUri))
      }
    } catch (e) {
      throw e;
    }
  }
  
  
  async function runModel() {
    try {
      const inputData = await preProcess()
      const feeds: Record<any, any> = {};
      feeds[myModel.inputNames[0]] = inputData;
      const fetches = await myModel.run(feeds);
      const output = fetches[myModel.outputNames[0]];

      if (!output) {
        console.log("Model ran unsuccessfully")
      } else {
        const outputArray = output.data as Float32Array
        await postProcess(Array.from(outputArray));
      }
    } catch (e) {
      throw e;
    }
  }


  async function preProcess(){
    await draw();

    floatPixelsY.forEach((value, index) => {

      const currentIndex = index * 4;
      const red = bitmapPixel[currentIndex]
      const green = bitmapPixel[currentIndex + 1]
      const blue = bitmapPixel[currentIndex + 2]

      floatPixelsY[index] = pixelsRGBToYCbCr(red, green, blue, "y")
    })

    cbArray.forEach((value, index) => {

      const currentIndex = index * 4;
      const red = bitmapScaledPixel[currentIndex]
      const green = bitmapScaledPixel[currentIndex + 1]
      const blue = bitmapScaledPixel[currentIndex + 2]
      
      cbArray[index] = pixelsRGBToYCbCr(red, green, blue, "cb")
      crArray[index] = pixelsRGBToYCbCr(red, green, blue, "cr")
    })
    let tensor = new ortWeb.Tensor(floatPixelsY, [1, 1, imageDim, imageDim])
    return tensor
  }


  async function postProcess(outputArray: number []) {
    outputArray.forEach((value, index) => {

      const pixel = pixelsYCbCrToRGB(value, cbArray[index], crArray[index], platform)
      const currentIndex = index * 4;
      const data = myImageScaledData.data

      data[currentIndex] = pixel[0]
      data[currentIndex + 1] = pixel[1]
      data[currentIndex + 2] = pixel[2]
    })

    const canvas = document.getElementById("canvas") as HTMLCanvasElement
    const ctx = canvas.getContext("2d")
    
    if (ctx && kdv) {  
      kdv.putImageData(myImageScaledData, 0, 0);
      kdv.save();
      ctx.drawImage(offscreen, 0, 0, scaledImageDim, scaledImageDim, 0, 0, 350, 350)
      ctx.save()
    }
  }


  async function draw() {
    const image1 = document.getElementById('selectedImage') as HTMLImageElement
    kdv = offscreen.getContext('2d')

    if (kdv) {
      kdv.drawImage(image1, 0, 0, imageDim, imageDim)
      const myImageData = kdv.getImageData(0, 0, imageDim, imageDim)
      bitmapPixel = Array.from(myImageData.data)
      kdv.clearRect(0, 0, imageDim, imageDim)

      kdv.drawImage(image1, 0, 0, scaledImageDim, scaledImageDim)
      myImageScaledData = kdv.getImageData(0, 0, scaledImageDim, scaledImageDim)
      bitmapScaledPixel = Array.from(myImageScaledData.data)
      kdv.clearRect(0, 0, scaledImageDim, scaledImageDim)
    }
  }
    

  let openImagePickerAsync = async () => {
    let permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      alert("Permission to access camera roll is required!");
    }

    let pickerResult = await ImagePicker.launchImageLibraryAsync();

    if (pickerResult.cancelled === true) {
      return;
    }

    const imageResult = await ImageManipulator.manipulateAsync(
      pickerResult.uri, [
        {resize: {height: imageDim, width: imageDim}}
      ]
    )
    setSelectedImage({ localUri: imageResult.uri });
  };


  return (
    <View style={[styles.containerWeb, ] }>
      <Text style={styles.instructions}>
        To upload a photo, press the button!
      </Text>
      {selectedImage != null &&
      <View style={styles.imageView}>
        <Image
        source={{ uri: selectedImage.localUri }}
        style={styles.thumbnail}
      />
          
        {selectedImage != null && 
          <canvas id='canvas' width="350" height="350">
              
              <img id='selectedImage' src={selectedImage.localUri} width="250" height="250" alt='' />
              
          </canvas>

        }
      </View>}

      <TouchableOpacity onPress={openImagePickerAsync} >
        <Button
        title="Pick a photo <3"
        onPress={openImagePickerAsync}
        color="#118ab2"
        />

      </TouchableOpacity>

      <Button
        title='Run Model'
        onPress={runModel}
        color="#118ab2"
        />

    </View>
  );
}
