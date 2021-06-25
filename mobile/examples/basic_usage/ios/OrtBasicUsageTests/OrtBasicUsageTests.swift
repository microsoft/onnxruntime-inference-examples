// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import XCTest

@testable import OrtBasicUsage

class OrtBasicUsageTests: XCTestCase {
  func testAddSwift() throws {
    XCTAssertEqual(try SwiftOrtAdd(1.0, 2.0), 3.0, accuracy: 1e-4)
  }

  func testAddObjc() throws {
    XCTAssertEqual(try ObjcOrtBasicUsage.add(2.0, 3.0).floatValue, 5.0, accuracy: 1e-4)
  }
}
