import SwiftUI

enum ModelTesterError: Error {
  case runtimeError(msg: String)
}

enum ExecutionProviderType: String, CaseIterable, Identifiable {
  case cpu = "CPU"
  case coreml = "CoreML"

  var id: Self { self }
}

struct ContentView: View {
  @State private var runResultMessage: String = ""
  @State private var isRunning: Bool = false
  @State private var numIterations: UInt = 10
  @State private var executionProviderType: ExecutionProviderType = .cpu

  private func Run() {
    isRunning = true

    DispatchQueue.global().async {
      var output: String
      do {
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx") else {
          throw ModelTesterError.runtimeError(msg: "Failed to find model file path.")
        }

        let config = ModelRunnerRunConfig()
        config.setModelPath(modelPath)
        config.setNumIterations(numIterations)

        if executionProviderType != .cpu {
          config.setExecutionProvider(executionProviderType.rawValue)
        }

        output = try ModelRunner.run(config: config)
      } catch {
        output = "Error: \(error)"
      }

      print(output)
      runResultMessage = output
      isRunning = false
    }
  }

  var body: some View {
    Form {
      Text("Iterations:")
      TextField(
        "", value: $numIterations,
        format: IntegerFormatStyle<UInt>.number
      ).keyboardType(.numberPad)

      Picker("Execution provider type", selection: $executionProviderType) {
        ForEach(ExecutionProviderType.allCases) { epType in
          Text(epType.rawValue).tag(epType)
        }
      }

      Button(action: Run) { Text("Run") }
        .disabled(isRunning)

      Text(runResultMessage)
        .font(.body.monospaced())
    }
  }
}

#Preview{
  ContentView()
}
