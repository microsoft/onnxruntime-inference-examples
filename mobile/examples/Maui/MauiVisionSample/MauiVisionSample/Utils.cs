// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using SkiaSharp;

namespace MauiVisionSample
{
    internal static class Utils
    {
        internal static async Task<byte[]> LoadResource(string name)
        {
            using Stream fileStream = await FileSystem.Current.OpenAppPackageFileAsync(name);
            using MemoryStream memoryStream = new MemoryStream();
            fileStream.CopyTo(memoryStream);
            return memoryStream.ToArray();
        }

        public static byte[] HandleOrientation(byte[] image)
        {
            using var memoryStream = new MemoryStream(image);
            using var imageData = SKData.Create(memoryStream);
            using var codec = SKCodec.Create(imageData);
            var orientation = codec.EncodedOrigin;

            using var bitmap = SKBitmap.Decode(image);
            using var adjustedBitmap = AdjustBitmapByOrientation(bitmap, orientation);

            // encode the raw bytes in a known format that SKBitmap.Decode can handle.
            // doing this makes our APIs a little more flexible as they can take multiple image formats as byte[].
            // alternatively we could use SKBitmap instead of byte[] to pass the data around and avoid some
            // SKBitmap.Encode/Decode calls, at the cost of being tightly coupled to the SKBitmap type.
            using var stream = new MemoryStream();
            using var wstream = new SKManagedWStream(stream);

            adjustedBitmap.Encode(wstream, SKEncodedImageFormat.Jpeg, 100);
            var bytes = stream.ToArray();

            return bytes;
        }

        static SKBitmap AdjustBitmapByOrientation(SKBitmap bitmap, SKEncodedOrigin orientation)
        {
            switch (orientation)
            {
                case SKEncodedOrigin.BottomRight:

                    using (var canvas = new SKCanvas(bitmap))
                    {
                        canvas.RotateDegrees(180, bitmap.Width / 2, bitmap.Height / 2);
                        canvas.DrawBitmap(bitmap.Copy(), 0, 0);
                    }

                    return bitmap;

                case SKEncodedOrigin.RightTop:

                    using (var rotatedBitmap = new SKBitmap(bitmap.Height, bitmap.Width))
                    {
                        using (var canvas = new SKCanvas(rotatedBitmap))
                        {
                            canvas.Translate(rotatedBitmap.Width, 0);
                            canvas.RotateDegrees(90);
                            canvas.DrawBitmap(bitmap, 0, 0);
                        }

                        rotatedBitmap.CopyTo(bitmap);
                        return bitmap;
                    }

                case SKEncodedOrigin.LeftBottom:

                    using (var rotatedBitmap = new SKBitmap(bitmap.Height, bitmap.Width))
                    {
                        using (var canvas = new SKCanvas(rotatedBitmap))
                        {
                            canvas.Translate(0, rotatedBitmap.Height);
                            canvas.RotateDegrees(270);
                            canvas.DrawBitmap(bitmap, 0, 0);
                        }

                        rotatedBitmap.CopyTo(bitmap);
                        return bitmap;
                    }

                default:
                    return bitmap;
            }
        }
    }
}

