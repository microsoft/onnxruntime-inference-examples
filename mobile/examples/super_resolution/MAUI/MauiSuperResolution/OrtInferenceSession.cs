using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;

namespace MauiSuperResolution
{
    public enum ExecutionProviders
    {
        CPU,    // CPU execution provider is always available by default
        NNAPI,  // NNAPI is available on Android
        CoreML, // CoreML is available on iOS/macOS
        XNNPACK // XNNPACK is available on ARM/ARM64 platforms and benefits 32-bit float models
    }

    // An inference session runs an ONNX model
    class OrtInferenceSession
    {
        public OrtInferenceSession(ExecutionProviders provider = ExecutionProviders.CPU)
        {
            _sessionOptions = new SessionOptions();
            switch (provider)
            {
                case ExecutionProviders.CPU:
                    break;
                case ExecutionProviders.NNAPI:
                    _sessionOptions.AppendExecutionProvider_Nnapi();
                    break;
                case ExecutionProviders.CoreML:
                    _sessionOptions.AppendExecutionProvider_CoreML();
                    break;
                case ExecutionProviders.XNNPACK:
                    _sessionOptions.AppendExecutionProvider("XNNPACK");
                    break;
            }

            // enable pre/post processing custom operators from onnxruntime-extensions
            _sessionOptions.RegisterOrtExtensions();
        }

        // async task to create the inference session which is an expensive operation.
        public async Task Create()
        {
            // create the InferenceSession. this is an expensive operation so only do this when necessary.
            // the InferenceSession supports multiple calls to Run, including concurrent calls.
            var modelBytes = await Utils.LoadResource("RealESRGAN_with_pre_post_processing.onnx");

            _inferenceSession = new InferenceSession(modelBytes, _sessionOptions);
        }

        public byte[] Run(byte[] jpgOrPngBytes)
        {
            // wrap the image bytes in a tensor
            var tensor = new DenseTensor<byte>(new Memory<byte>(jpgOrPngBytes), new[] { jpgOrPngBytes.Length });

            // create model input. the input name 'image' is defined in the model
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("image", tensor) };

            // Run inference
            var stopwatch = new Stopwatch();
            stopwatch.Start();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _inferenceSession.Run(inputs);
            stopwatch.Stop();
            _lasRunTimeMs = stopwatch.ElapsedMilliseconds;

            var output = results.First().AsEnumerable<byte>().ToArray();

            return output;
        }

        public long LastRunTimeMs => _lasRunTimeMs;

        private SessionOptions _sessionOptions;
        private InferenceSession _inferenceSession;
        private long _lasRunTimeMs = 0;
    }
}
