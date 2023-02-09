// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


import XCTest

@testable import ORTSuperResolution

final class ORTSuperResolutionTests: XCTestCase {

    func testPerformSuperResolution() throws {
        
        let outputImage = try ORTSuperResolutionPerformer.performSuperResolution()
        XCTAssertNotNil(outputImage, "check the output UIImage is not nil")
        XCTAssertEqual(outputImage.size.height, 672.0)
        XCTAssertEqual(outputImage.size.width, 672.0)

    }
}
