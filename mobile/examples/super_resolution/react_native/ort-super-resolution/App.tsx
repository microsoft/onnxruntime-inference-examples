import { Buffer } from "buffer";
import React, { useState } from 'react';
import { Alert, Button, StyleSheet, Text, View, Image } from 'react-native';
import { Asset } from 'expo-asset';
import { StatusBar } from 'expo-status-bar';
import * as FileSystem from 'expo-file-system';

import * as ort from 'onnxruntime-react-native';

let base64js = require('base64-js')
let myModel: ort.InferenceSession;

const getUint8ArrayFromUri = async (uri: string): Promise<Uint8Array> => {
  try {
    const fileContent = await FileSystem.readAsStringAsync(uri, {
      encoding: FileSystem.EncodingType.Base64,
    });
    const buffer = Buffer.from(fileContent, 'base64');
    return new Uint8Array(buffer);
  } catch (error) {
    console.error(error);
    return new Uint8Array();
  }
};

export default function App() {
  const [outputImage, setOutputImage] = React.useState<any>(null);

  async function loadModel() {
    try {
      const assets = await Asset.loadAsync(require('./assets/pytorch_superresolution_with_pre_post_processing_opset18.onnx'));
      const modelUri = assets[0].localUri;
      if (!modelUri) {
        Alert.alert('failed to get model URI', `${assets[0]}`);
      } else {
        myModel = await ort.InferenceSession.create(modelUri);
        Alert.alert(
          'model loaded successfully',
          `input names: ${myModel.inputNames}, output names: ${myModel.outputNames}`);
      }
    } catch (e) {
      Alert.alert('failed to load model', `${e}`);
      throw e;
    }
  }

  async function runModel() {
    try {
      const feeds: Record<string, ort.Tensor> = {};
      const assets = await Asset.loadAsync(require('./assets/cat_224x224.png'));
      let myImage = assets[0].localUri as string;
      const imageUint8ArrayData = await getUint8ArrayFromUri(myImage);
      const dataLength = imageUint8ArrayData.length;
      feeds[myModel.inputNames[0]] = new ort.Tensor(imageUint8ArrayData, [dataLength]);
      const fetches = await myModel.run(feeds);
      const output = fetches[myModel.outputNames[0]];    // output pixelbuffer

      let encodedString = base64js.fromByteArray(output.data);
      console.log(encodedString);

      setOutputImage({ localUri: `data:image/png;base64,${encodedString}` })

      if (!output) {
        Alert.alert('failed to get output', `${myModel.outputNames[0]}`);
      } else {
        Alert.alert('model inference successfully');
      }
    } catch (e) {
      Alert.alert('failed to inference model', `${e}`);
      throw e;
    }
  }


  return (
    <View style={styles.container}>
      <Text style={styles.heading}>ORT SuperResolution for React Native</Text>
      <Image
        source={require('./assets/cat_224x224.png')}
        style={styles.image}
        resizeMode="contain"></Image>
      <View>
        <Button title='Load model' onPress={loadModel}></Button>
        <Button title='Run' onPress={runModel}></Button>
      </View>
      <View>
        {
          outputImage != null &&
          <Image
            source={{ uri: outputImage.localUri }}
            style={styles.image2}
            resizeMode='contain'></Image>
        }
      </View>
      <StatusBar style="auto" />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  image: {
    flex: 1,
    position: 'absolute',
    top: 130,
    width: 200,
    height: 200,
  },
  image2: {
    position: 'absolute',
    marginLeft: -100,
    top: 50,
    width: 200,
    height: 200,
  },
  heading: {
    position: 'absolute',
    top: 5,
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    marginTop: 50,
    backgroundColor: 'gray',
    alignItems: 'flex-start',
  }
});

