// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VisionSample
{
    // See: https://github.com/onnx/models/tree/master/vision/classification/resnet#model
    // Model download: https://github.com/onnx/models/blob/master/vision/classification/resnet/model/resnet50-v1-7.onnx
    public class ResNetSample : VisionSampleBase<ResNetImageProcessor>
    {
        public const string Identifier = "ResNet50 v2";
        public const string ModelFilename = "resnet50.onnx";

        public ResNetSample()
            : base(Identifier, ModelFilename) {}

        protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
        {
            using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image)).ConfigureAwait(false);
            var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
            var predictions = await Task.Run(() => GetPredictions(tensor)).ConfigureAwait(false);
            var preprocessedImageData = await Task.Run(() => ImageProcessor.GetBytesForBitmap(preprocessedImage)).ConfigureAwait(false);

            var caption = string.Empty;

            if (predictions.Any())
            {
                var builder = new StringBuilder();

                if (predictions.Any())
                    builder.Append($"Top {predictions.Count} predictions: {Environment.NewLine}{Environment.NewLine}");

                foreach (var prediction in predictions)
                    builder.Append($"{prediction.Label} ({prediction.Confidence}){Environment.NewLine}");

                caption = builder.ToString();
            }

            return new ImageProcessingResult(preprocessedImageData, caption);
        }

        List<ResNetPrediction> GetPredictions(Tensor<float> input)
        {
            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("data", input) };

            // Run inference
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);

            // Postprocess to get softmax vector
            IEnumerable<float> output = results.First().AsEnumerable<float>();
            float sum = output.Sum(x => (float)Math.Exp(x));
            IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);

            // Extract top 10 predicted classes
            IEnumerable<ResNetPrediction> top10 = softmax
                .Select((x, i) => new ResNetPrediction
                {
                    Label = ResNetLabelMap.Labels[i],
                    Confidence = x
                })
                .OrderByDescending(x => x.Confidence)
                .Take(10);

            return top10.ToList();
        }
    }
}
