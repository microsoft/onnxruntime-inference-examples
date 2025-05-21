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
  enum Field: Hashable {
    case numIterations
    case executionProviderType
    case executionProviderOptionsText
  }

  @State private var runResultMessage: String = ""
  @State private var isRunning: Bool = false
  @State private var numIterations: UInt = 10
  @State private var executionProviderType: ExecutionProviderType = .cpu
  @State private var executionProviderOptionsText: String = ""
  @FocusState private var focusedField: Field?

  private func parseExecutionProviderOptionsText() throws -> [String: String] {
    let executionProviderOptions =
      try executionProviderOptionsText
      .components(separatedBy: "\n")
      .reduce(
        into: [String: String](),
        { options, line in
          guard !line.isEmpty else {
            return
          }
          let nameAndValue = line.split(separator: ":", maxSplits: 2)
          guard nameAndValue.count == 2 else {
            throw ModelTesterError.runtimeError(msg: "Failed to parse provider option: '\(line)'")
          }
          options[String(nameAndValue[0])] = String(nameAndValue[1])
        })

    return executionProviderOptions
  }

  private func run() {
    isRunning = true
    focusedField = nil

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
          let executionProviderOptions = try parseExecutionProviderOptionsText()

          print("Execution provider type: \(executionProviderType)")
          print("Execution provider options: \(executionProviderOptions)")
          config.setExecutionProvider(executionProviderType.rawValue, options: executionProviderOptions)
        }

        output = try ModelRunner.run(config: config)
      } catch ModelTesterError.runtimeError(let msg) {
        output = "Error: \(msg)"
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
      Text("Iterations")
      TextField(
        "", value: $numIterations,
        format: IntegerFormatStyle<UInt>.number
      ).keyboardType(.numberPad)
        .focused($focusedField, equals: .numIterations)

      Picker("Execution provider type", selection: $executionProviderType) {
        ForEach(ExecutionProviderType.allCases) { epType in
          Text(epType.rawValue).tag(epType)
        }
      }.focused($focusedField, equals: .executionProviderType)

      if executionProviderType != .cpu {
        Text("Execution provider options")
        Text("The expected format is 'name:value', one per line")
          .font(.caption)
        TextEditor(text: $executionProviderOptionsText)
          .focused($focusedField, equals: .executionProviderOptionsText)
      }

      Button(action: run) { Text("Run") }
        .disabled(isRunning)

      Text(runResultMessage)
        .font(.body.monospaced())
    }
  }
}

#Preview{
  ContentView()
}
