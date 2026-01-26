using Contoso.ML.OnnxRuntime.EP.Basic;
using Microsoft.ML.OnnxRuntime;

class Program
{
    static void Main()
    {
        string epLibPath = BasicEp.GetLibraryPath();
        string epRegistrationName = "basic_ep_registration";
        string epName = BasicEp.GetEpName();

        var env = OrtEnv.Instance();
        env.RegisterExecutionProviderLibrary(epRegistrationName, epLibPath);
        Console.WriteLine($"Registered EP library: {epLibPath}");

        try
        {
            // Find the OrtEpDevice for the EP
            OrtEpDevice? epDevice = null;
            foreach (var d in env.GetEpDevices())
            {
                if (string.Equals(epName, d.EpName, StringComparison.OrdinalIgnoreCase))
                {
                    epDevice = d;
                }
            }

            if (epDevice == null)
            {
                Console.Error.WriteLine($"ERROR: Unable to find OrtEpDevice with name {epName}");
                return;
            }
            Console.WriteLine($"Found OrtEpDevice for EP: {epName}");

            // Create session with EP
            using var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider(env, [epDevice], new Dictionary<string, string> { });
            sessionOptions.AddSessionConfigEntry("session.disable_cpu_ep_fallback", "1");  // Don't run on CPU EP

            string inputModelPath = Path.Combine(AppContext.BaseDirectory, "mul.onnx");
            Console.WriteLine($"Loading model: {inputModelPath}");

            using var session = new InferenceSession(inputModelPath, sessionOptions);

            // Run model
            float[] inputData = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f];
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory<float>(inputData, [2, 3]);
            var inputValues = new List<OrtValue> { inputOrtValue, inputOrtValue }.AsReadOnly();
            var inputNames = new List<string> { "x", "y" }.AsReadOnly();
            using var runOptions = new RunOptions();

            using var outputs = session.Run(runOptions, inputNames, inputValues, session.OutputNames);

            Console.WriteLine($"Input: {string.Join(", ", inputData)}");
            Console.WriteLine($"Output: {string.Join(", ", outputs[0].GetTensorDataAsSpan<float>().ToArray())}");
        }
        finally
        {
            env.UnregisterExecutionProviderLibrary(epRegistrationName);
        }
    }
}
