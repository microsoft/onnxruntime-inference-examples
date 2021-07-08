// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import SwiftUI

struct ContentView: View {
  private let audioRecorder = AudioRecorder()
  private let speechRecognizer = try! SpeechRecognizer()

  @State private var message: String = ""
  @State private var successful: Bool = true

  @State private var readyToRecord: Bool = true

  private func recordAndRecognize() {
    audioRecorder.record { recordResult in
      let recognizeResult = recordResult.flatMap { recordingBufferAndData in
        return speechRecognizer.evaluate(inputData: recordingBufferAndData.data)
      }
      endRecordAndRecognize(recognizeResult)
    }
  }

  private func endRecordAndRecognize(_ result: Result<String, Error>) {
    DispatchQueue.main.async {
      switch result {
      case .success(let transcription):
        message = transcription
        successful = true
      case .failure(let error):
        message = "Error: \(error)"
        successful = false
      }
      readyToRecord = true
    }
  }

  var body: some View {
    VStack {
      Text("Press \"Record\", say something, and get recognized!")
        .padding()

      Button("Record") {
        readyToRecord = false
        recordAndRecognize()
      }
      .padding()
      .disabled(!readyToRecord)

      Text("\(message)")
        .foregroundColor(successful ? .none : .red)
        .padding()
    }
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
