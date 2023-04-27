// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime;

namespace MauiVisionSample
{
    // See: https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/superresolution_e2e.py
    // Model does pre and post processing so we just need to feed it an image and it will return an image 
    public class SuperResolutionSample : VisionSampleBase<SuperResolutionImageProcessor>
    {
        public const string Identifier = "SuperResolution";
        public const string ModelFilename = "pytorch_superresolution.with_pre_post_processing.onnx";

        public SuperResolutionSample() : base(Identifier, ModelFilename) {}

        protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
        {
            // Convert to Tensor by wrapping image bytes
            var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(image))
                                   .ConfigureAwait(false);

            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("image", tensor) };

            // Run inference
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);

            var output = results.First().AsEnumerable<byte>().ToArray();

            return new ImageProcessingResult(output, "Image with super resolution applied.");
        }
    }
}
