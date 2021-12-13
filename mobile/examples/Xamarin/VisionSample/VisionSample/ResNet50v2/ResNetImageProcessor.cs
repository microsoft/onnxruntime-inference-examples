// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace VisionSample
{
    public class ResNetImageProcessor : SkiaSharpImageProcessor<ResNetPrediction, float>
    {
        const int RequiredHeight = 224;
        const int RequiredWidth = 224;

        protected override SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage)
        {
            float ratio = (float)Math.Min(RequiredWidth, RequiredHeight) / Math.Min(sourceImage.Width, sourceImage.Height);
            using SKBitmap scaledBitmap = sourceImage.Resize(new SKImageInfo((int)(Math.Ceiling(ratio * sourceImage.Width)), (int)(Math.Ceiling(ratio * sourceImage.Height))), SKFilterQuality.Medium);

            var horizontalCrop = Math.Max(scaledBitmap.Width - RequiredWidth, 0);
            var verticalCrop = Math.Max(scaledBitmap.Height - RequiredHeight, 0);
            var leftOffset = horizontalCrop == 0 ? 0 : horizontalCrop / 2;
            var topOffset = verticalCrop == 0 ? 0 : verticalCrop / 2;

            var cropRect = SKRectI.Create(new SKPointI(leftOffset, topOffset), new SKSizeI(RequiredWidth, RequiredHeight));

            using SKImage currentImage = SKImage.FromBitmap(scaledBitmap);
            using SKImage croppedImage = currentImage.Subset(cropRect);
            SKBitmap croppedBitmap = SKBitmap.FromImage(croppedImage);

            return croppedBitmap;
        }

        protected override Tensor<float> OnGetTensorForImage(SKBitmap image)
        {
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, RequiredHeight, RequiredWidth });

            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image.GetPixel(x, y);
                    input[0, 0, y, x] = ((pixel.Red / 255f) - mean[0]) / stddev[0];
                    input[0, 1, y, x] = ((pixel.Green / 255f) - mean[1]) / stddev[1];
                    input[0, 2, y, x] = ((pixel.Blue / 255f) - mean[2]) / stddev[2];
                }
            }

            return input;
        }
    }
}