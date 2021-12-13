// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VisionSample
{
    public interface IImageProcessor<TImage, TPrediction, TTensor>
    {
        TImage PreprocessSourceImage(byte[] sourceImage);
        Tensor<TTensor> GetTensorForImage(TImage image);
        byte[] ApplyPredictionsToImage(IList<TPrediction> predictions, TImage image);
    }
}