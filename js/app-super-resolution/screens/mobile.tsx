// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


import { StatusBar } from 'expo-status-bar';
import React, { useState } from 'react';
import { Alert, Button, Text, View, NativeModules, Image, ScrollView, ToastAndroid, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import { MainScreenProps } from './NavigStack';
import { openImagePicker, converter, loadModelAll, runModelAll } from '../misc/utilities';
import { styles } from '../misc/styles';


const bitmapModule = NativeModules.Bitmap
const imageDim = 224
const scaledImageDim = imageDim * 3
const platform = Platform.OS


let ort: any
if (platform == "android") { ort = require("onnxruntime-react-native") }

let isLoaded = false;
let model: any;
let floatPixelsY = new Float32Array(imageDim * imageDim)
let cbArray = new Float32Array(scaledImageDim*scaledImageDim)
let crArray = new Float32Array(scaledImageDim*scaledImageDim)
let bitmapPixel: number[] = Array(imageDim * imageDim);
let bitmapScaledPixel: number[] = Array(scaledImageDim * scaledImageDim);


export default function AndroidApp({ navigation, route }: MainScreenProps) {
  const [selectedImage, setSelectedImage] = useState<any>(null);
  const [outputImage, setOutputImage] = useState<any>(null);
  const [myModel, setModel] = useState(model);


  async function openImagePickerAsync() {
    const pickerResult = await openImagePicker() as string
    await imageToPixel(pickerResult)
    return
  };


  async function imageToPixel(uri: string) {
    const imageResult = await ImageManipulator.manipulateAsync(
      uri, [
      { resize: { height: imageDim, width: imageDim } }
    ]
    )

    const imageScaled = await ImageManipulator.manipulateAsync(
      uri, [
      { resize: { height: scaledImageDim, width: scaledImageDim } }
    ]
    )

    bitmapPixel = await bitmapModule.getPixels(imageResult.uri).then(

      (image: any) => {
        return Array.from(image.pixels);
      }
    )

    bitmapScaledPixel = await bitmapModule.getPixels(imageScaled.uri).then(
      (image: any) => {
        return Array.from(image.pixels);
      }
    )

    setSelectedImage({
      localUri: imageResult.uri,
    });

    setOutputImage(null)
  }


  async function openCameraAsync() {
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();

    if (permissionResult.granted === false) {
      alert("Permission to access Camera Roll is Required!");
      return;
    }

    const pickerResult = await ImagePicker.launchCameraAsync({ allowsEditing: true });

    if (pickerResult.cancelled === true) {
      return;
    }

    await imageToPixel(pickerResult.uri)
  }


  async function preprocess() {

    const result = await converter([bitmapPixel, bitmapScaledPixel], "YCbCR", platform) as Float32Array[]
    floatPixelsY = result[0]
    cbArray = result[1]
    crArray = result[2]

    let tensor = new ort.Tensor(floatPixelsY, [1, 1, imageDim, imageDim])
    return tensor
  };


  async function postprocess(floatArray: number[]) {

    const intArray = await converter([floatArray, Array.from(cbArray), Array.from(crArray)], "RGB", platform) as any[]

    let imageUri = await bitmapModule.getImageUri(Array.from(intArray)).then(
      (image: any) => {
        return image.uri
      }
    )

    const imageRotated = await ImageManipulator.manipulateAsync(imageUri, [
      { rotate: 90 },
      { flip: ImageManipulator.FlipType.Horizontal }
    ])

    setOutputImage({ localUri: imageRotated.uri })
  };


  async function loadModel() {
    try {
      const model = await loadModelAll(ort)
      setModel(model)

    } catch (e) {
      Alert.alert('failed to load model', `${e}`);
      throw e;
    }
  }


  async function runModel() {
    try {

      const inputData = await preprocess()
      const output = await runModelAll(ort, inputData, myModel)
      if(output) await postprocess(output)
      ToastAndroid.show('SUPER_RESOLUTION DONE\n  SWYPE DOWN', ToastAndroid.LONG)

    } catch (e) {
      Alert.alert('failed to inference model', `${e}`);
      throw e;
    }
  };

  if (!isLoaded || !myModel) {
    loadModel().then(() => {
      isLoaded = true;
    })

  }


  return (
    <View style={styles.containerAndroid}>
      <Text style={styles.item}>Using ONNX Runtime in React Native to perform Super Resolution on Images</Text>
      <View style={styles.userInput}>
        <Button title='Upload Image' onPress={openImagePickerAsync} color="#219ebc" />
        <Button title='Open Camera' onPress={openCameraAsync} color="#219ebc" />
      </View>
      {
        selectedImage !== null &&
        <ScrollView style={styles.scrollView}>
          <Image
            source={{ uri: selectedImage.localUri }}
            style={styles.thumbnail}
          />
          {
            outputImage !== null &&
            <Image
              source={{ uri: outputImage.localUri }}
              style={styles.thumbnail}
            />
          }
        </ScrollView>}
      {isLoaded && selectedImage !== null &&
        <View style={styles.userInput}>
          <Button title='Process Image' onPress={runModel} color="#219ebc" />
        </View>
      }


      <StatusBar style="auto" />
    </View>
  );
};
