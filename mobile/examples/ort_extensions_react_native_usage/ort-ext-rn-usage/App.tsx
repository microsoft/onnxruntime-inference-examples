import { Buffer } from "buffer";
import { Alert, Button, StyleSheet, Text, View } from 'react-native';
import { Asset } from 'expo-asset';
import { StatusBar } from 'expo-status-bar';
import * as FileSystem from 'expo-file-system';

import * as ort from 'onnxruntime-react-native';

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

async function loadModel() {
  try {
    const assets = await Asset.loadAsync(require('./assets/decode_image.onnx'));
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
    const assets = await Asset.loadAsync(require('./assets/r32_g64_b128_32x32.png'));
    const imageUint8ArrayData = await getUint8ArrayFromUri((assets[0].localUri) as string);
    const dataLength = imageUint8ArrayData.length;
    feeds[myModel.inputNames[0]] = new ort.Tensor(imageUint8ArrayData, [dataLength]);
    const fetches = await myModel.run(feeds);
    const output = fetches[myModel.outputNames[0]];
    if (!output) {
      Alert.alert('failed to get output', `${myModel.outputNames[0]}`);
    } else {
      Alert.alert(
        'model inference successfully',
        `output shape: ${output.dims}, output data: ${output.data}`);
    }
  } catch (e) {
    Alert.alert('failed to inference model', `${e}`);
    throw e;
  }
}

export default function App() {
  return (
    <View style={styles.container}>
      <Text>using ONNX Runtime with extensions for React Native</Text>
      <Button title='Load model' onPress={loadModel}></Button>
      <Button title='Run' onPress={runModel}></Button>
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
});
