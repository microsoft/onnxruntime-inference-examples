// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation

private let kTokens = [
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
    guard let modelPath = Bundle.main.path(forResource: "wav2vec2-base-960h.all", ofType: "ort") else {
      throw SpeechRecognizerError.Error("Failed to find model file.")
    }
    ortSession = try ORTSession(env: ortEnv, modelPath: modelPath, sessionOptions: nil)
  }

  private func postprocess(modelOutput: UnsafeBufferPointer<Float>) -> String {
    func maxIndex(_ values: UnsafeBufferPointer<Float>) -> Int? {
      var max: (idx: Int, value: Float)?
      for (idx, value) in values.enumerated() {
        if max == nil || value > max!.value {
          max = (idx, value)
        }
      }
      return max?.idx
    }

    func tokenIndexToOutput(_ index: Int) -> String {
      if index == 4 {
        return " "
      } else if index > 4 && index < kTokens.count {
        return kTokens[index]
      }
      return ""
    }

    let n = modelOutput.count / kTokens.count
    var resultTokenIndices: [Int] = []

    for i in 0..<n {
      let tokenValues = UnsafeBufferPointer<Float>(rebasing: modelOutput[i * kTokens.count..<(i + 1) * kTokens.count])
      if let tokenIndex = maxIndex(tokenValues) {
        // append without consecutive duplicates
        if tokenIndex != resultTokenIndices.last {
          resultTokenIndices.append(tokenIndex)
        }
      }
    }

    return resultTokenIndices.map(tokenIndexToOutput).joined()
  }

  func evaluate(inputData: Data) throws -> String {
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