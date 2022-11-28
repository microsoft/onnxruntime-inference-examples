// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


import { StatusBar } from 'expo-status-bar';
import React, { useState } from 'react';
import { Alert, Button, Text, View, NativeModules, Image, ScrollView, ToastAndroid, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
//import * as ImageManipulator from 'expo-image-manipulator';
import { MainScreenProps } from './NavigStack';
import { /*converter,*/ loadModelAll, runModelAll } from '../misc/utilities';
import { styles } from '../misc/styles';
//import { btoa, atob } from 'react-native-quick-base64';


let base64ImageStringFromUpload = ""
let imageHeight = 1
let imageWidth = 1
let ort: any
const platform = Platform.OS
if (platform == "android") { ort = require("onnxruntime-react-native") }
let base64js = require('base64-js')
let isLoaded = false;
let model: any;

/** 
 * Note: Code snippet used for imageToPixel(). Commented out here as it's not needed with updated
 * data pre/post processing support.
 * 
const bitmapModule = NativeModules.Bitmap
const imageDim = 224
const scaledImageDim = imageDim * 3
let floatPixelsY = new Float32Array(imageDim * imageDim)
let cbArray = new Float32Array(scaledImageDim*scaledImageDim)
let crArray = new Float32Array(scaledImageDim*scaledImageDim)
let bitmapPixel: number[] = Array(imageDim * imageDim);
let bitmapScaledPixel: number[] = Array(scaledImageDim * scaledImageDim);

*/

export default function AndroidApp({ navigation, route }: MainScreenProps) {
  const [selectedImage, setSelectedImage] = useState<any>(null);
  const [outputImage, setOutputImage] = useState<any>(null);
  const [myModel, setModel] = useState(model);

  /**
   * Opens up the library of the mobile device in order to select an image from the library.
   */
  async function openImagePickerAsync() {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      alert("Permission to access Camera Roll is Required!");
      return;
    }

    // enable return base64 encoded data option when calling ImagePicker
    const pickerResult = await ImagePicker.launchImageLibraryAsync({ allowsEditing: true, base64: true });

    if (pickerResult.cancelled === true) {
      return;
    }

    base64ImageStringFromUpload = pickerResult.base64;
    imageHeight = pickerResult.height;
    imageWidth = pickerResult.width;
    // Check the input image size currently there's a restriction from superRes model to take in 224x224 image
    // TODO: Delete when restrictions are removed
    console.log("Height of input jpeg:" + imageHeight);
    console.log("Width of input jpeg:" + imageWidth);

    setSelectedImage({
      localUri: pickerResult.uri
    });

    setOutputImage(null)
  };

  /**
   * It generates the hex pixel data of an image given its source.
   * It firstly resizes the image to the right dimensions, then makes use of an 
   * Android [Native Module](https://reactnative.dev/docs/next/native-modules-android) to get a [height x width] array containing the pixel data. 

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
  */

  // Note: only updated the path for opening from ImagePicker for now.
  // TODO: update path for opening from camera
  /**
   * Opens up the camera of the mobile device to take a picture.
   */
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

    //await imageToPixel(pickerResult.uri)
  }

  function _base64ToByteArray(base64ImageStringFromUpload: string) {
    let decodedByteArray = base64js.toByteArray(base64ImageStringFromUpload)
    return decodedByteArray
  };

  /**
   * Prepare the input data and input tensor for model
   * @returns ort tensor type
   */
  async function preprocess() {
    let inputDataArray = _base64ToByteArray(base64ImageStringFromUpload)
    let dataLength = inputDataArray.length
    let tensor = new ort.Tensor(inputDataArray, [dataLength])
    return tensor
  };

  /**
   * Process output of inference result and set output image
   * @param OutputArr ort inferenceSession call result
   */
  async function postprocess(OutputArr: number[]) {
    let encodedString = base64js.fromByteArray(OutputArr)
    setOutputImage({ localUri: `data:image/jpeg;base64,${encodedString}` })
  };

  /**
   * Loads ORT model on mobile
   */
  async function loadModel() {
    try {
      const model = await loadModelAll(ort)
      setModel(model)

    } catch (e) {
      Alert.alert('failed to load model', `${e}`);
      throw e;
    }
  }

  /**
   * Runs ORT model on mobile
   */
  async function runModel() {
    try {
      const inputData = await preprocess()
      const output = await runModelAll(inputData, myModel)
      if (output) await postprocess(output)
      ToastAndroid.show('SUPER_RESOLUTION DONE\n  SWIPE DOWN', ToastAndroid.LONG)
    } catch (e) {
      Alert.alert('failed to inference model', `${e}`);
      throw e;
    }
  };

  // Automatically loads the model immediately the screen is rendered
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
