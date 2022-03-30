// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation

enum SwiftOrtBasicUsageError: Error {
  case error(_ message: String)
}

// Some helper functions to copy values out of/into a Data instance.

private func dataCopiedFromArray<T>(_ array: [T]) -> Data {
  return array.withUnsafeBufferPointer { buffer -> Data in
    return Data(buffer: buffer)
  }
}

private func arrayCopiedFromData<T>(_ data: Data) -> [T]? {
  guard data.count % MemoryLayout<T>.stride == 0 else { return nil }
  return data.withUnsafeBytes { bytes -> [T] in
    return Array(bytes.bindMemory(to: T.self))
  }
}

/// Adds `a` and `b` using ONNX Runtime.
func SwiftOrtAdd(_ a: Float, _ b: Float) throws -> Float {
  // We will run a simple model which adds two floats.
  // The inputs are named `A` and `B` and the output is named `C` (A + B = C).
  // All inputs and outputs are float tensors with shape [1].
  let modelPath = try ObjcOrtBasicUsage.getAddModelPath()

  // First, we create the ORT environment.
  // The environment is required in order to create an ORT session.
  // ORTLoggingLevel.warning should show us only important messages.
  let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)

  // Next, we will create some ORT values for our input tensors. We have two
  // floats, `a` and `b`.
  let valueDataType = ORTTensorElementDataType.float
  let valueShape: [NSNumber] = [1]

  // `aData` will hold the memory of the input ORT value.
  // We set it to contain a copy of our input float, `a`.
  let aData = dataCopiedFromArray([a])
  // This will create a value with a tensor with the a copy of `a`'s data, of
  // type float, and with shape [1].
  let aInputValue = try ORTValue(
    tensorData: NSMutableData(data: aData),
    elementType: valueDataType,
    shape: valueShape)

  // And we do the same for `b`.
  let bData = dataCopiedFromArray([b])
  let bInputValue = try ORTValue(
    tensorData: NSMutableData(data: bData),
    elementType: valueDataType,
    shape: valueShape)

  // Now, we will create an ORT session to run our model.
  // One can configure session options with a session options object
  // (ORTSessionOptions).
  // We use the default options with sessionOptions: nil.
  let session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: nil)

  // A session provides methods to get the model input and output names.
  // We will confirm that they are what we expect.
  let inputNames = try session.inputNames()
  precondition(inputNames.sorted() == ["A", "B"])
  let outputNames = try session.outputNames()
  precondition(outputNames.sorted() == ["C"])

  // With a session and input values, we have what we need to run the model.
  // We provide a dictionary mapping from input name to value and a set of
  // output names.
  // This run method will run the model, allocating the output(s), and return
  // them in a dictionary mapping from output name to value.
  // As with session creation, it is possible to configure run options with a
  // run options object (ORTRunOptions).
  // We use the default options with runOptions: nil.
  let outputs = try session.run(
    withInputs: ["A": aInputValue, "B": bInputValue],
    outputNames: ["C"],
    runOptions: nil)

  // After running the model, we will get the output.
  guard let cValue = outputs["C"] else {
    throw SwiftOrtBasicUsageError.error("failed to get model output")
  }

  // We will query the output value's tensor type and shape.
  // This method may be called if the value is a tensor. We know it is.
  let cTypeAndShape = try cValue.tensorTypeAndShapeInfo()

  // We will confirm that the shape is what we expect.
  guard cTypeAndShape.shape == [1] else {
    throw SwiftOrtBasicUsageError.error("output does not have expected size")
  }

  // Finally, we will access the output value's data.
  let cData = try cValue.tensorData() as Data

  // Since we called run without pre-allocated outputs, ORT owns the output
  // values. We must not access an output value's memory after it is
  // deinitialized. So, we will copy the data here.
  guard let cArr: [Float32] = arrayCopiedFromData(cData) else {
    throw SwiftOrtBasicUsageError.error("failed to copy output data")
  }

  return cArr[0]
}
