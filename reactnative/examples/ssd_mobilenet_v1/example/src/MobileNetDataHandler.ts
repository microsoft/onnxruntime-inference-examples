import { NativeModules } from 'react-native';
import type { InferenceSession } from 'onnxruntime-reactnative';

export interface MobileNetInput {
  [name: string]: {
    dims: number[];
    type: string;
    data: string; // encoded tensor data
  };
}

export interface MobileNetOutput {
  [name: string]: {
    data: string; // encoded tensor data
  };
}

export interface MobileNetResult {
  result: string;
}

type MobileNetType = {
  getLocalModelPath(): Promise<string>;
  getImagePath(): Promise<string>;
  preprocess(uri: string): Promise<MobileNetInput>;
  postprocess(result: MobileNetOutput): Promise<MobileNetResult>;
  run(
    uri: string,
    fetches: InferenceSession.FetchesType,
    options: InferenceSession.RunOptions
  ): Promise<MobileNetResult>;
};

const MobileNet = NativeModules.MobileNetDataHandler;

export default MobileNet as MobileNetType;
