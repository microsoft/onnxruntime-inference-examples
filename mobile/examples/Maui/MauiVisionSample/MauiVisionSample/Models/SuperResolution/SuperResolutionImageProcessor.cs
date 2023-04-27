// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MauiVisionSample
{
    // See: https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/superresolution_e2e.py
    public class SuperResolutionImageProcessor : IImageProcessor<byte[], byte[], byte>
    {
        public byte[] ApplyPredictionsToImage(IList<byte[]> predictions, byte[] orig_image)
        {
            // predictions are the updated image produced by the model
            return predictions[0];
        }

        public Tensor<byte> GetTensorForImage(byte[] image)
        {
            // create Tensor from image bytes
            return new DenseTensor<byte>(new Memory<byte>(image), new[] { image.Length });
        }

        public byte[] PreprocessSourceImage(byte[] sourceImage)
        {
            // done by the model
            return sourceImage;
        }
    }
}
