using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.PixelFormats;

namespace image_classification {
    class Program {
        static void Main (string[] args) {
            string modelFilePath = args[0];
            string imageFilePath = args[1];

            // Read the image
            Console.WriteLine ("Reading image: " + imageFilePath);
            Image<Rgb24> image = Image.Load<Rgb24> (imageFilePath, out IImageFormat format);
            int width = image.Width;
            int height = image.Height;

            // Convert the image to a tensor
            Tensor<byte> input = new DenseTensor<byte> (new [] { height, width, 3 });
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        input[y, x, 0] = pixelSpan[x].R;
                        input[y, x, 1] = pixelSpan[x].G;
                        input[y, x, 2] = pixelSpan[x].B;
                    }
                }
            });

            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("image", input)
            };

            using var session = new InferenceSession(modelFilePath);
            
            foreach (var r in  session.Run(inputs))
            {
                Console.WriteLine("Output for {0}", r.Name);
                if (r.Name == "top_classes") {
                    Console.WriteLine(r.AsTensor<Int64>().GetArrayString());
                } else {
                    Console.WriteLine(r.AsTensor<float>().GetArrayString());
                }
            }

        }

    }
}