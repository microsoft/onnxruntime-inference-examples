using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace image_classification
{
    class Program {
        static void Main (string[] args) {
            string modelFilePath = args[0];
            string imageFilePath = args[1];

            using var runOptions = new RunOptions();
            using var session = new InferenceSession(modelFilePath);

            // Read the image
            Console.WriteLine ($"Reading image: {imageFilePath}");
            using Image<Rgb24> image = Image.Load<Rgb24> (imageFilePath, out IImageFormat format);
            int width = image.Width;
            int height = image.Height;

            Console.WriteLine($"Image: W:{width} H:{height} loaded");

            // As this library does not allow access to its content directly, we need to copy it to a new tensor
            // within the OrtValue. This must be disposed of when no longer needed. We use default CPU based allocator
            // here and we are copying straight to the native memory. In cases when the data resides in an array, we
            // can use it directly without copying by using CreateTensorValueFromMemory().

            // The below allocates an internal native buffer.
            // Alternatively, we can also allocate a managed buffer, copy data there and then map it to the OrtValue.
            // No significant difference, since one copy would have to be made, but may be important to have access
            // to a managed buffer from C#.
            using var ortValue = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, 
                TensorElementType.UInt8, new long[] { height, width, 3 });

            image.ProcessPixelRows(accessor =>
            {
                int flatIndex = 0;
                var destSpan = ortValue.GetTensorMutableDataAsSpan<byte>();
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    Debug.Assert(pixelSpan.Length == accessor.Width);

                    // We can copy this byte by byte or we can copy the entire row.
                    // Because we know that Rgb24 is a Sequential Layout structure that consists
                    // of exactly 3 bytes.
                    var byteSpan = MemoryMarshal.Cast<Rgb24, byte>(pixelSpan);
                    Debug.Assert(byteSpan.Length == accessor.Width * 3);
                    var destSlice = destSpan.Slice(flatIndex, byteSpan.Length);
                    byteSpan.CopyTo(destSlice);
                    flatIndex += byteSpan.Length;

                    // Byte by byte copy
                    //for (int x = 0; x < accessor.Width; x++)
                    //{
                    //    destSpan[flatIndex++] = pixelSpan[x].R;
                    //    destSpan[flatIndex++] = pixelSpan[x].G;
                    //    destSpan[flatIndex++] = pixelSpan[x].B;
                    //}
                }
            });


            var inputs = new Dictionary<string, OrtValue>
            {
                { "image", ortValue }
            };

            // Outputs are disposable OrtValues, must be disposed of.
            using var outputs = session.Run(runOptions, inputs, session.OutputNames);

            for(int i = 0; i < outputs.Count; ++i)
            {
                Console.WriteLine($"Output for {i} {session.OutputNames[i]}");
                if (session.OutputNames[i] == "top_classes") {
                    // Direct access to native output buffer
                    var dataSpan = outputs[i].GetTensorDataAsSpan<Int64>();
                    foreach (var v in dataSpan)
                        Console.WriteLine(v);
                } else {
                    // Direct access to native output buffer
                    var dataSpan = outputs[i].GetTensorDataAsSpan<float>();
                    foreach (var v in dataSpan)
                        Console.WriteLine(v);
                }
                Console.WriteLine();
            }

        }

    }
}