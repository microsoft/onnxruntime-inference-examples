import { StatusBar } from 'expo-status-bar';
import { Alert, Button, StyleSheet, Text, View } from 'react-native';

import * as ort from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';

// Note: These modules are used for reading model into bytes
// import RNFS from 'react-native-fs';
// import base64 from 'base64-js';

let myModel: ort.InferenceSession;

async function loadModel() {
  try {
    // Note: `.onnx` model files can be viewed in Netron (https://github.com/lutzroeder/netron) to see
    // model inputs/outputs detail and data types, shapes of those, etc.
    const assets = await Asset.loadAsync(require('./assets/mnist.onnx'));
    const modelUri = assets[0].localUri;
    if (!modelUri) {
      Alert.alert('failed to get model URI', `${assets[0]}`);
    } else {
      // load model from model url path
      myModel = await ort.InferenceSession.create(modelUri);
      Alert.alert(
        'model loaded successfully',
        `input names: ${myModel.inputNames}, output names: ${myModel.outputNames}`);

      // loading model from bytes
      // const base64Str = await RNFS.readFile(modelUri, 'base64');
      // const uint8Array = base64.toByteArray(base64Str);
      // myModel = await ort.InferenceSession.create(uint8Array);
    }
  } catch (e) {
    Alert.alert('failed to load model', `${e}`);
    throw e;
  }
}

async function runModel() {
  try {
    // Prepare model input data
    // Note: In real use case, you must set the inputData to the actual input values
    const inputData = new Float32Array(28 * 28);
    const feeds:Record<string, ort.Tensor> = {};
    feeds[myModel.inputNames[0]] = new ort.Tensor(inputData, [1, 1, 28, 28]);
    // Run inference session
    const fetches = await myModel.run(feeds);
    // Process output
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
      <Text>ONNX Runtime React Native Basic Usage</Text>
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
