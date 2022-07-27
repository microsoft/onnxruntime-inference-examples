// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace MauiVisionSample
{
    public class SkiaSharpImageProcessor<TPrediction, TTensor> : IImageProcessor<SKBitmap, TPrediction, TTensor>
    {
        protected virtual SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage) => sourceImage;
        protected virtual Tensor<TTensor> OnGetTensorForImage(SKBitmap image) => throw new NotImplementedException();
        protected virtual void OnPrepareToApplyPredictions(SKBitmap image, SKCanvas canvas) { }
        protected virtual void OnApplyPrediction(TPrediction prediction, SKPaint textPaint, SKPaint rectPaint, SKCanvas canvas) { }

        public byte[] ApplyPredictionsToImage(IList<TPrediction> predictions, SKBitmap image)
        {
            // Annotate image to reflect predictions and save for viewing
            using SKSurface surface = SKSurface.Create(new SKImageInfo(image.Width, image.Height));
            using SKCanvas canvas = surface.Canvas;

            // Normalize paint size based on 800f shortest edge
            float ratio = 800f / Math.Min(image.Width, image.Height);
            var textSize = 32 * ratio;
            var strokeWidth = 2f * ratio;

            using SKPaint textPaint = new SKPaint { TextSize = textSize, Color = SKColors.White };
            using SKPaint rectPaint = new SKPaint { StrokeWidth = strokeWidth, IsStroke = true, Color = SKColors.Red };

            canvas.DrawBitmap(image, 0, 0);

            OnPrepareToApplyPredictions(image, canvas);

            foreach (var prediction in predictions)
                OnApplyPrediction(prediction, textPaint, rectPaint, canvas);

            canvas.Flush();

            using var snapshot = surface.Snapshot();
            using var imageData = snapshot.Encode(SKEncodedImageFormat.Jpeg, 100);
            byte[] bytes = imageData.ToArray();

            return bytes;
        }

        public byte[] GetBytesForBitmap(SKBitmap bitmap)
        {
            using var image = SKImage.FromBitmap(bitmap);
            using var data = image.Encode(SKEncodedImageFormat.Jpeg, 100);
            var bytes = data.ToArray();

            return bytes;
        }

        public Tensor<TTensor> GetTensorForImage(SKBitmap image)
            => OnGetTensorForImage(image);

        public Size GetSizeForSourceImage(byte[] sourceImage)
        {
            using var image = SKBitmap.Decode(sourceImage);
            return new Size(image.Width, image.Height);
        }

        public SKBitmap GetImageFromBytes(byte[] sourceImage, float shortestEdge = -1.0f)
        {
            var image = SKBitmap.Decode(sourceImage);

            if (shortestEdge > 0.0)
            {
                float ratio = shortestEdge / Math.Min(image.Width, image.Height);
                image = image.Resize(new SKImageInfo((int)(ratio * image.Width), 
                                                     (int)(ratio * image.Height)), 
                                                     SKFilterQuality.Medium);
            }

            return image;
        }

        public SKBitmap PreprocessSourceImage(byte[] sourceImage)
        {
            // Read image
            using var image = SKBitmap.Decode(sourceImage);
            return OnPreprocessSourceImage(image);
        }
    }
}