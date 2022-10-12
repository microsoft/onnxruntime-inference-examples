// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation

// these labels correspond to the model's output values
// the labels and postprocessing logic were copied and adapted from:
// https://github.com/pytorch/ios-demo-app/blob/f2b9aa196821c136d3299b99c5dd592de1fa1776/SpeechRecognition/create_wav2vec2.py#L10
private let kLabels = [
  "<s>", "<pad>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H", "S", "R", "D", "L", "U", "M", "W", "C", "F",
  "G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q", "Z",
]

class SpeechRecognizer {
  private let ortEnv: ORTEnv
  private let ortSession: ORTSession

  enum SpeechRecognizerError: Error {
    case Error(_ message: String)
  }

  init() throws {
    ortEnv = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
    guard let modelPath = Bundle.main.path(forResource: "wav2vec2-base-960h", ofType: "ort") else {
      throw SpeechRecognizerError.Error("Failed to find model file.")
    }
    ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: nil)
  }

  private func postprocess(modelOutput: UnsafeBufferPointer<Float>) -> String {
    func maxIndex<S>(_ values: S) -> Int? where S: Sequence, S.Element == Float {
      var max: (idx: Int, value: Float)?
      for (idx, value) in values.enumerated() {
        if max == nil || value > max!.value {
          max = (idx, value)
        }
      }
      return max?.idx
    }

    func labelIndexToOutput(_ index: Int) -> String {
      if index == 4 {
        return " "
      } else if index > 4 && index < kLabels.count {
        return kLabels[index]
      }
      return ""
    }

    precondition(modelOutput.count % kLabels.count == 0)
    let n = modelOutput.count / kLabels.count
    var resultLabelIndices: [Int] = []

    for i in 0..<n {
      let labelValues = modelOutput[i * kLabels.count..<(i + 1) * kLabels.count]
      if let labelIndex = maxIndex(labelValues) {
        // append without consecutive duplicates
        if labelIndex != resultLabelIndices.last {
          resultLabelIndices.append(labelIndex)
        }
      }
    }

    return resultLabelIndices.map(labelIndexToOutput).joined()
  }

  func evaluate(inputData: Data) -> Result<String, Error> {
    return Result<String, Error> { () -> String in
      let inputShape: [NSNumber] = [1, inputData.count / MemoryLayout<Float>.stride as NSNumber]
      let input = try ORTValue(
        tensorData: NSMutableData(data: inputData),
        elementType: ORTTensorElementDataType.float,
        shape: inputShape)

      let startTime = DispatchTime.now()
      let outputs = try ortSession.run(
        withInputs: ["input": input],
        outputNames: ["output"],
        runOptions: nil)
      let endTime = DispatchTime.now()
      print("ORT session run time: \(Float(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1.0e6) ms")

      guard let output = outputs["output"] else {
        throw SpeechRecognizerError.Error("Failed to get model output.")
      }

      let outputData = try output.tensorData() as Data
      let result = outputData.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) -> String in
        let floatBuffer = buffer.bindMemory(to: Float.self)
        return postprocess(modelOutput: floatBuffer)
      }

      print("result: '\(result)'")
      return result
    }
  }
}
