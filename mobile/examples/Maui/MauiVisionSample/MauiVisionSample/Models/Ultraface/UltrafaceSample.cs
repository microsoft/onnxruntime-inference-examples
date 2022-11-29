// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MauiVisionSample
{
    // See: https://github.com/onnx/models/tree/master/vision/body_analysis/ultraface#model
    // Model download: https://github.com/onnx/models/blob/master/vision/body_analysis/ultraface/models/version-RFB-320.onnx
    public class UltrafaceSample : VisionSampleBase<UltrafaceImageProcessor>
    {
        public const string Identifier    = "Ultraface";
        public const string ModelFilename = "Ultraface_version-RFB-320.onnx";

        public UltrafaceSample()
            : base(Identifier, ModelFilename) {}

        protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
        {
            // do initial resize maintaining the aspect ratio so the smallest size is 800. this is arbitrary and 
            // chosen to be a good size to dispay to the user with the results
            using var sourceImage = await Task.Run(() => ImageProcessor.GetImageFromBytes(image, 800f))
                                              .ConfigureAwait(false);

            // do the preprocessing to resize the image to the 320x240 with the model expects. 
            // NOTE: this does not maintain the aspect ratio but works well enough with this particular model.
            //       it may be better in other scenarios to resize and crop to convert the original image to a
            //       320x240 image.
            using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image))
                                                    .ConfigureAwait(false);

            // Convert to Tensor of normalized float RGB data with NCHW ordering
            var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage))
                                   .ConfigureAwait(false);


            // Run the model
            var predictions = await Task.Run(() => GetPredictions(tensor, sourceImage.Width, sourceImage.Height))
                                        .ConfigureAwait(false);

            // Draw the bounding box for the best prediction on the image from the first resize. 
            var outputImage = await Task.Run(() => ImageProcessor.ApplyPredictionsToImage(predictions, sourceImage))
                                        .ConfigureAwait(false);

            return new ImageProcessingResult(outputImage);
        }

        List<UltrafacePrediction> GetPredictions(Tensor<float> input, int sourceImageWidth, int sourceImageHeight)
        {
            // Setup inputs. Names used must match the input names in the model. 
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", input) };

            // Run inference
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);

            // Process the results. 
            //   First result is the confidence score for each match
            //   Second result are the values to draw a bounding box for each match
            //
            // Note that the correct processing is always model specific. Things like the format of the values for
            // the bounding boxes can vary by model.
            var resultsArray = results.ToArray();
            float[] confidences = resultsArray[0].AsEnumerable<float>().ToArray();
            float[] boxes = resultsArray[1].AsEnumerable<float>().ToArray();

            // Confidences are represented by 2 values - the second is for the face and the first is ignored
            var scores = confidences.Where((val, index) => index % 2 == 1).ToList();

            // If there were no good matches we return an empty prediction
            if (!scores.Any(i => i < 0.5))
            {
                return new List<UltrafacePrediction>(); ;
            }

            // find the best score
            float highestScore = scores.Max(); 
            var indexForHighestScore = scores.IndexOf(highestScore);
            var boxOffset = indexForHighestScore * 4;

            return new List<UltrafacePrediction> 
                   { 
                        new UltrafacePrediction
                        {
                            Confidence = scores[indexForHighestScore],
                            Box = new PredictionBox(
                                boxes[boxOffset + 0] * sourceImageWidth,
                                boxes[boxOffset + 1] * sourceImageHeight,
                                boxes[boxOffset + 2] * sourceImageWidth,
                                boxes[boxOffset + 3] * sourceImageHeight)
                        }
                   };
        }
    }
}
