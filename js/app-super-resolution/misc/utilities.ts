// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.


import * as ImagePicker from 'expo-image-picker';
import { Asset } from 'expo-asset';
type PlatformTypes = "ios" | "android" | "windows" | "macos" | "web"


export function pixelsRGBToYCbCr(red: number, green: number, blue: number, mode: string): number {

    let result = 0
    if (mode == "y") {
        result  = (0.299 * red +
                    0.587 * green +
                    0.114 * blue) / 255

    }else if (mode == "cb"){
        result = ((-0.168935) * red +
                    (-0.331665) * green +
                    0.50059 * blue) + 128
                    
    }else if (mode == "cr") {
        result = ((0.499813 * red +
            (-0.418531) * green +
            (-0.081282) * blue) + 128)

    }
    
    return result
    
}


export function pixelsYCbCrToRGB(pixel: number, cb: number, cr: number, platform: PlatformTypes) {
    const y = Math.min(Math.max((pixel * 255), 0), 255);

    const red = Math.min(Math.max((y + (1.4025 * (cr-0x80))), 0), 255);

    const green = Math.min(Math.max((y + ((-0.34373) * (cb-0x80)) +
                                          ((-0.7144) * (cr-0x80))), 0), 255);

    const blue = Math.min(Math.max((y + (1.77200 * (cb-0x80))), 0), 255);

    if (platform == "web") {
        const pixels = Array.of(red, green, blue)
        return pixels

    }else if (platform == "android") {
        const intPixel =  Array.of(
                        ((0xFF << 24) |
                        ((0xFF & blue) << 16) |
                        ((0xFF & green) << 8) |
                        ((0xFF & red)))
                        )

        return intPixel
    }else return Array(0)
}


export async function openImagePicker() {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (permissionResult.granted === false) {
      alert("Permission to access Camera Roll is Required!");
      return "0";
    }

    const pickerResult = await ImagePicker.launchImageLibraryAsync({ allowsEditing: true });

    if (pickerResult.cancelled === true) {
      return pickerResult.cancelled;
    }
    return pickerResult.uri

}


export async function converter(array: number[][], mode: "YCbCR"|"RGB", platform: PlatformTypes) {
    const imageDim = 224
    const scaledDim = imageDim * 3

    if (mode == "YCbCR"){
        const floatPixelsY = new Float32Array(imageDim*imageDim)
        const cbArray = new Float32Array(scaledDim*scaledDim)
        const crArray = new Float32Array(scaledDim*scaledDim)
        const [inputArray, scaledArray] = [array[0], array[1]]

        for (let i = 0; i < imageDim*imageDim; i++) {
            let red = 0
            let green = 0
            let blue = 0

            if (platform == "android") {
                const value = inputArray[i]
                red = (value >> 16 & 0xFF)
                green = (value >> 8 & 0xFF)
                blue = (value & 0xFF)
            }else if(platform == "web") {
                const currIndex = i * 4;
                red = inputArray[currIndex]
                green = inputArray[currIndex + 1]
                blue = inputArray[currIndex + 2]
            };

            floatPixelsY[i] = pixelsRGBToYCbCr(red, green, blue, "y")
        }

        for (let i = 0; i < scaledDim*scaledDim; i++) {
            let red = 0
            let green = 0
            let blue = 0
        
            if (platform == "android") {
                const value = scaledArray[i]
                red = (value >> 16 & 0xFF)
                green = (value >> 8 & 0xFF)
                blue = (value & 0xFF)
            }else if(platform == "web") {
                const currIndex = i * 4;
                red = scaledArray[currIndex]
                green = scaledArray[currIndex + 1]
                blue = scaledArray[currIndex + 2]
            };
            cbArray[i] = pixelsRGBToYCbCr(red, green, blue, "cb")
            crArray[i] = pixelsRGBToYCbCr(red, green, blue, "cr")
        }
        return Array.of(floatPixelsY, cbArray, crArray)       
    } 
    else if (mode == "RGB"){
        const outputArray = array[0]
        const cbArray = array[1]
        const crArray = array[2]

        let intArray = platform == "android"? new Array(scaledDim*scaledDim): new Array(scaledDim*scaledDim*4)

        for (let i=0; i < scaledDim*scaledDim; i++) {
            if (platform == "android") {
                intArray[i] = pixelsYCbCrToRGB(outputArray[i], cbArray[i], crArray[i], platform)[0]

            }else if (platform == "web") {
                const pixel = pixelsYCbCrToRGB(outputArray[i], cbArray[i], crArray[i], platform)
                const currIndex = i * 4;

                intArray[currIndex] = pixel[0]
                intArray[currIndex + 1] = pixel[1]
                intArray[currIndex + 2] = pixel[2]
                intArray[currIndex + 3] = 255
            }
        }

        return intArray
    }
}


export async function loadModelAll(ort: any) {
    const assets = await Asset.loadAsync(require('../assets/super_resnet12.ort'));
    const modelUri = assets[0].localUri;

    if (!modelUri) {
        console.log("Model loaded unsuccessfully")
    }
    else{
        const model = await ort.InferenceSession.create(modelUri);
        return model
    }
}


export async function runModelAll(ort: any, inputData: any, model: any) {
    const feeds: Record<any, any> = {};
    feeds[model.inputNames[0]] = inputData;
    const fetches = await model.run(feeds);
    const output = fetches[model.outputNames[0]];

    if (!output) {
      console.log("Model ran unsuccessfully")
    } else {
      const outputArray = output.data as Float32Array
      return Array.from(outputArray);
    }
}
