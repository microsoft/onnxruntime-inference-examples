import SwiftUI

enum ModelTesterError: Error {
  case runtimeError(msg: String)
}

struct ContentView: View {
  @State private var runResultMessage: String = ""
  @State private var isRunning: Bool = false
  @State private var numIterations: UInt32 = 10

  private func Run() {
    isRunning = true

    DispatchQueue.global().async {
      var output: String
      do {
        guard let modelPath = Bundle.main.path(forResource: "mobilenetv2-12", ofType: "onnx") else {
          throw ModelTesterError.runtimeError(msg: "Failed to find model file path.")
        }

        output = try ModelRunner.run(
          withModelPath: modelPath,
          numIterations: numIterations)
      } catch {
        output = "Error: \(error)"
      }

      print(output)
      runResultMessage = output
      isRunning = false
    }
  }

  var body: some View {
    VStack {
      HStack {
        Text("Iterations:")
        TextField(
          "", value: $numIterations,
          format: IntegerFormatStyle<UInt32>.number
        )
        .keyboardType(.numberPad)
      }

      Button(action: Run) { Text("Run") }
        .disabled(isRunning)

      Text(runResultMessage)
        .font(.body.monospaced())
    }
    .padding()
  }
}

#Preview{
  ContentView()
}
