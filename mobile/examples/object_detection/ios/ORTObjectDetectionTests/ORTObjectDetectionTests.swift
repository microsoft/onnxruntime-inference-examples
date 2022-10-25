// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import XCTest

@testable import ORTObjectDetection

class ORTObjectDetectionTests: XCTestCase {
  func testModelLoads() throws {
    let modelHandler = ModelHandler(
      modelFileInfo: (name: "ssd_mobilenet_v1", extension: "ort"),
      labelsFileInfo: (name: "labelmap", extension: "txt"))
    XCTAssertNotNil(modelHandler)
  }
}
