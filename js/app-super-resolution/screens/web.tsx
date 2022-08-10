// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


import { Button, Text, View, Image, TouchableOpacity } from 'react-native';
import { styles } from '../misc/styles';
import React, {useState, useEffect, useRef} from 'react';
import { Camera } from 'expo-camera';
import { Platform } from 'expo-modules-core';
import * as ImageManipulator from 'expo-image-manipulator';
import { MainScreenProps } from './NavigStack';
import { openImagePicker, converter, loadModelAll, runModelAll } from '../misc/utilities';


const platform = Platform.OS

let model: any;
let ort: any;

if (platform == "web") {ort = require("onnxruntime-web")}

let isLoaded = false;
const imageDim = 224
const scaledImageDim = imageDim * 3

let offscreen: any 
if (platform == "web") {offscreen = new OffscreenCanvas(1000, 1000)}
let myImageScaledData: ImageData
let kdv: OffscreenCanvasRenderingContext2D | null
let ctx: CanvasRenderingContext2D |null
let floatPixelsY = new Float32Array(imageDim * imageDim)
let cbArray = new Float32Array(scaledImageDim * scaledImageDim)
let crArray = new Float32Array(scaledImageDim * scaledImageDim)
let bitmapPixel: number[] = Array(imageDim*imageDim);
let bitmapScaledPixel: number[] = Array(scaledImageDim*scaledImageDim);


export default function WebApp({navigation, route}: MainScreenProps) {
  const [selectedImage, setSelectedImage] = useState<any>(null);
  const [myModel, setModel] = useState(model);
  const [myCamera, setCamera] = useState(false);
  const [hasPermission, setHasPermission] = useState<any>(null);
  const ref = useRef<any>(null)
  

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);


  let openImagePickerAsync = async () => {
    setCamera(false)
    const pickerResult = await openImagePicker() as string
    const imageResult = await ImageManipulator.manipulateAsync(
      pickerResult, [
        {resize: {height: imageDim, width: imageDim}}
      ]
    )
    setSelectedImage({ localUri: imageResult.uri });
    const canvas = document.getElementById("canvas") as HTMLCanvasElement
    ctx = canvas.getContext("2d")
    if (ctx) {
      ctx.clearRect(0, 0, 350, 350)
    }
  };


  async function _takePhoto() {
    const photo = await ref.current.takePictureAsync({isImageMirror: true})
    const[big, small] = [Math.max(photo.height, photo.width), Math.min(photo.height, photo.width)]

    const imageResult = await ImageManipulator.manipulateAsync(
      photo.uri, [
        {crop:{height: small, width: small, originX: Math.floor((big-small)/2), originY: 0}},
        {resize: {height: imageDim, width: imageDim}}
      ]
    )

    setSelectedImage({ localUri: imageResult.uri });
    setCamera(false);
    const canvas = document.getElementById("canvas") as HTMLCanvasElement
    ctx = canvas.getContext("2d")
    if (ctx) {
      ctx.clearRect(0, 0, 350, 350)
    }
  } 


  async function preProcess(){
    await draw();

    const result = await converter([bitmapPixel, bitmapScaledPixel], "YCbCR", platform) as Float32Array[]
    floatPixelsY = result[0]
    cbArray = result[1]
    crArray = result[2]

    let tensor = new ort.Tensor(floatPixelsY, [1, 1, imageDim, imageDim])
    return tensor
  }


  async function postProcess(outputArray: number []) {
    
    const newImageData = await converter(Array.of(outputArray, Array.from(cbArray), Array.from(crArray)), "RGB", platform) as any[]

    let data = myImageScaledData.data

    newImageData.forEach((value, index) => {
      data[index] = value
    })

    
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
  

  async function loadModel() {
    try {
      const model = await loadModelAll(ort)
      setModel(model)
      
    } catch (e) {
      throw e;
    }
  }
  
  
  async function runModel() {
    try {
      const inputData = await preProcess()
      const output = await runModelAll(ort, inputData, myModel)
      if(output) await postProcess(output)
    } catch (e) {
      throw e;
    }
  }

    
  if (!isLoaded || !myModel) {

    loadModel().then(() => {
      isLoaded = true;
    })
  } 

  return (
    <View style={[styles.containerWeb, ] }>
      <Text style={styles.instructions}>
        Pick an Image or Take a Picture!
      </Text>
      {myCamera && hasPermission &&
      <Camera style={{ flex: 1, alignSelf: 'stretch' }} ref={ref}>
        <View 
          style={{
            flex: 1,
            backgroundColor: 'transparent',
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'flex-end',
          }}>

          <TouchableOpacity onPress={_takePhoto}>
            <Image
              source={require('../assets/cabut.png')}
              style={styles.cameraButton}
            />
          </TouchableOpacity>
        </View>
      </Camera>
      }

      {selectedImage != null &&
      <View style={styles.imageView}>
        <Image
          source={{ uri: selectedImage.localUri }}
          style={styles.thumbnail}
        />
        <canvas id='canvas' width="350" height="350">
          <img id='selectedImage' src={selectedImage.localUri} width="250" height="250" alt='' />
        </canvas>
      </View>}
      <View style={styles.userInput}>
        <Button
          title="Pick a photo <3"
          onPress={openImagePickerAsync}
          color="#118ab2"
        />
        <Button
          title="Take a photo <3"
          onPress={()=> {
            setSelectedImage(null);
            setCamera(true);
          }}
          color="#118ab2"
      />
      </View>

      <Button
        title='Run Model'
        onPress={runModel}
        color="#118ab2"
        />

    </View>
  );
}
