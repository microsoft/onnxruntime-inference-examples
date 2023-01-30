// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


import XCTest

@testable import ORTSuperResolution

final class ORTSuperResolutionTests: XCTestCase {

    func testPerformSuperResolution() throws {
        // test that it doesn't throw
        try ORTSuperResolutionPerformer.performSuperResolution()
    }
}
