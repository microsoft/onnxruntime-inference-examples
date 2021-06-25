// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import XCTest

@testable import SpeechRecognition

class SpeechRecognitionTests: XCTestCase {
  func testModelLoadsAndRuns() throws {
    let recognizer = try SpeechRecognizer()
    let dummyData = Data(count: 16000 * MemoryLayout<Float>.size)
    _ = try recognizer.evaluate(inputData: dummyData).get()
  }
}
