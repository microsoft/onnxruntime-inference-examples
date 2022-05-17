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
            Console.WriteLine ("Converting image to tensor");
            Tensor<float> input = new DenseTensor<float> (new [] { 1, 3, height, width });
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        input[0, 0, y, x] = pixelSpan[x].R;
                        input[0, 1, y, x] = pixelSpan[x].G;
                        input[0, 2, y, x] = pixelSpan[x].B;
                    }
                }
            });
            Console.WriteLine ("Tensor: " + input);

            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("image", input)
            };

            Console.WriteLine ("Creating session ...");
            using var session = new InferenceSession(modelFilePath);

            Console.WriteLine("InferenceSession: " + session);

            Console.WriteLine ("Running model " + modelFilePath);            
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            Console.WriteLine ("Image classification: " + results);
        }

    }
}