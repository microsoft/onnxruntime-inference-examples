// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import SwiftUI

private func EvaluateSum(_ a: Float, _ b: Float, _ c: Float) -> String {
  do {
    // Add using ORT from Objective-C.
    let aPlusB = try ObjcOrtBasicUsage.add(NSNumber(value: a), NSNumber(value: b)).floatValue

    // Add using ORT from Swift.
    let result = try SwiftOrtAdd(aPlusB, c)

    return String(result)
  } catch {
    return "Error: \(error)"
  }
}

struct ContentView: View {
  var body: some View {
    VStack {
      Text("Hello, world from ONNX Runtime!")
        .padding()

      let a: Float = 3
      let b: Float = 4
      let c: Float = 5

      Text("\(a) + \(b) + \(c) = \(EvaluateSum(a, b, c))")
        .padding()
    }
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
