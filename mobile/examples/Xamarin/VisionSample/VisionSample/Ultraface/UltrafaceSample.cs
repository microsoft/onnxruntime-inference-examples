// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VisionSample
{
    // See: https://github.com/onnx/models/tree/master/vision/body_analysis/ultraface#model
    // Model download: https://github.com/onnx/models/blob/master/vision/body_analysis/ultraface/models/version-RFB-320.onnx
    public class UltrafaceSample : VisionSampleBase<UltrafaceImageProcessor>
    {
        public const string Identifier = "Ultraface";
        public const string ModelFilename = "ultraface.onnx";

        public UltrafaceSample()
            : base(Identifier, ModelFilename) {}

        protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
        {
            using var sourceImage = await Task.Run(() => ImageProcessor.GetImageFromBytes(image, 800f)).ConfigureAwait(false);
            using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image)).ConfigureAwait(false);
            var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
            var predictions = await Task.Run(() => GetPredictions(tensor, sourceImage.Width, sourceImage.Height)).ConfigureAwait(false);
            var outputImage = await Task.Run(() => ImageProcessor.ApplyPredictionsToImage(predictions, sourceImage)).ConfigureAwait(false);

            return new ImageProcessingResult(outputImage);
        }

        List<UltrafacePrediction> GetPredictions(Tensor<float> input, int sourceImageWidth, int sourceImageHeight)
        {
            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", input) };

            // Run inference
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);

            // Postprocess
            var resultsArray = results.ToArray();
            float[] confidences = resultsArray[0].AsEnumerable<float>().ToArray();
            float[] boxes = resultsArray[1].AsEnumerable<float>().ToArray();

            // Confidences are represented by 2 values - the second is for the face
            var scores = confidences.Where((val, index) => index % 2 == 1).ToList();

            if (!scores.Any(i => i < 0.5))
                return new List<UltrafacePrediction>(); ;

            // find the best score
            float highestScore = scores.Max(); 
            var indexForHighestScore = scores.IndexOf(highestScore);
            var boxOffset = indexForHighestScore * 4;

            return new List<UltrafacePrediction> { new UltrafacePrediction
            {
                Confidence = scores[indexForHighestScore],
                Box = new PredictionBox(
                    boxes[boxOffset + 0] * sourceImageWidth,
                    boxes[boxOffset + 1] * sourceImageHeight,
                    boxes[boxOffset + 2] * sourceImageWidth,
                    boxes[boxOffset + 3] * sourceImageHeight)
            }};
        }
    }
}
