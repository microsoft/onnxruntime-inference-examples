// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MauiVisionSample
{
    // See: https://github.com/onnx/models/tree/main/vision/classification/mobilenet
    // Model download: https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx
    // NOTE: We use the fp32 version of the model in this example as the int8 version uses internal ONNX Runtime 
    // operators. Due to that, it will not work well with NNAPI or CoreML.
    public class MobilenetSample : VisionSampleBase<MobilenetImageProcessor>
    {
        public const string Identifier = "Mobilenet V2";
        public const string ModelFilename = "mobilenetv2-12.onnx";

        public MobilenetSample()
            : base(Identifier, ModelFilename) {}

        protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
        {
            // Resize and center crop
            using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image)).ConfigureAwait(false);

            // Convert to Tensor of normalized float RGB data with NCHW ordering
            var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);

            // Run the model
            var predictions = await Task.Run(() => GetPredictions(tensor)).ConfigureAwait(false);

            // Get the pre-processed image for display to the user so they can see the actual input to the model
            var preprocessedImageData = await Task.Run(() => ImageProcessor.GetBytesForBitmap(preprocessedImage)).ConfigureAwait(false);

            var caption = string.Empty;

            if (predictions.Any())
            {
                var builder = new StringBuilder();

                if (predictions.Any())
                {
                    builder.Append($"Top {predictions.Count} predictions: {Environment.NewLine}{Environment.NewLine}");
                }

                foreach (var prediction in predictions)
                {
                    builder.Append($"{prediction.Label} ({prediction.Confidence * 100:0.00}%){Environment.NewLine}");
                }

                caption = builder.ToString();
            }

            return new ImageProcessingResult(preprocessedImageData, caption);
        }

        List<MobilenetPrediction> GetPredictions(Tensor<float> input)
        {
            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", input) };

            // Run inference
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);

            // Postprocess to get softmax vector
            IEnumerable<float> output = results.First().AsEnumerable<float>();
            float sum = output.Sum(x => (float)Math.Exp(x));
            IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);

            // Extract top 3 predicted classes
            var top3 = softmax.Select((x, i) => new MobilenetPrediction
                                                {
                                                    Label = MobilenetLabelMap.Labels[i],
                                                    Confidence = x
                                                })
                              .OrderByDescending(x => x.Confidence)
                              .Take(3)
                              .ToList();

            return top3;
        }
    }
}
