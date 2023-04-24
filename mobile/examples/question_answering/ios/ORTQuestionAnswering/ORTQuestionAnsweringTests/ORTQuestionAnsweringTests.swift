// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import XCTest

@testable import ORTQuestionAnswering

final class ORTQuestionAnsweringTests: XCTestCase {

    func testPerformQuestionAnswering() throws {
        let inputText = "How long did it take for their digestion?"
        let inputContext = "Article: We are introduced to the narrator, a pilot, and his ideas about grown-ups. Once when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest. It was a picture of a boa constrictor in the act of swallowing an animal. Here is a copy of the drawing.In the book it said: 'Boa constrictors swallow their prey whole, without chewing it. After that they are not able to move, and they sleep through the six months that they need for digestion.'"
        let outputAnswer = try ORTQuestionAnsweringPerformer.performQuestionAnswering(inputText, context: inputContext)
        XCTAssertNotNil(outputAnswer, "check the output string is not nil")
        XCTAssertEqual(outputAnswer, "six months")
    }

}
