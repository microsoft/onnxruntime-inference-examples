/*
Copyright (C) 2021, Intel Corporation
SPDX-License-Identifier: Apache-2.0
*/

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Diagnostics;

namespace yolov3
{
    class Program
    {
        static void Main(string[] args)
        {
            // string is null or empty 
            if (args == null || args.Length < 3)
            {
                Console.WriteLine("Usage information: dotnet run model.onnx input.jpg output.jpg");
                return;
            } else
            {
                if (!(File.Exists(args[0])))
                {
                    Console.WriteLine("Model Path does not exist");
                    return;
                }
                if (!(File.Exists(args[1])))
                {
                    Console.WriteLine("Input Path does not exist");
                    return;
                }
            }

            // Read paths
            string modelFilePath = args[0];
            string imageFilePath = args[1];
            string outImageFilePath = args[2];

            // Session Options
            using SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            options.AppendExecutionProvider_OpenVINO(@"MYRIAD_FP16");
            options.AppendExecutionProvider_CPU(1);

            // Load the model
            using var session = new InferenceSession(modelFilePath, options);

            // Load the input image
            using Image imageOrg = Image.Load(imageFilePath);

            //Letterbox image
            var iw = imageOrg.Width;
            var ih = imageOrg.Height;
            var w = 416;
            var h = 416;

            if ((iw == 0) || (ih == 0))
            {
                Console.WriteLine("Math error: Attempted to divide by Zero");
                return;
            }

            float width = (float)w / iw;
            float height = (float)h / ih;

            float scale = Math.Min(width, height);

            var nw = (int)(iw * scale);
            var nh = (int)(ih * scale);

            var pad_dims_w = (w - nw) / 2;
            var pad_dims_h = (h - nh) / 2;

            // Resize image using default bicubic sampler 
            using var image = imageOrg.Clone(x => x.Resize((nw), (nh)));

            using var clone = new Image<Rgb24>(w, h);
            clone.Mutate(i => i.Fill(Color.Gray));
            clone.Mutate(o => o.DrawImage(image, new Point(pad_dims_w, pad_dims_h), 1f)); // draw the first one top left

            // Preprocessing image
            // We use DenseTensor for multi-dimensional access
            // There is a sample code doing multi-dim indexing w/o DenseTensor
            // using ShapeUtils.
            DenseTensor<float> input = new(new[] { 1, 3, h, w });
            clone.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < pixelSpan.Length; x++)
                    {
                        input[0, 0, y, x] = pixelSpan[x].B / 255f;
                        input[0, 1, y, x] = pixelSpan[x].G / 255f;
                        input[0, 2, y, x] = pixelSpan[x].R / 255f;
                    }
                }
            });

            // Pin tensor memory and use it directly
            // It will be unpinned on ortValue disposal
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                input.Buffer, new long[] { 1, 3, h, w });

            //Get the Image Shape
            float[] imageShape = { ih, iw }; // This will be pinned and used directly for input
            using var imageShapeOrtValue = OrtValue.CreateTensorValueFromMemory(imageShape, new long[] { 1, 2 });

            // Setup inputs, request all output names.
            var inputs = new Dictionary<string, OrtValue>
            {
                { "input_1", inputOrtValue },
                { "image_shape", imageShapeOrtValue }
            };

            using var runOptions = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> results = session.Run(runOptions, inputs, session.OutputNames);

            Console.WriteLine("Inference done");

            // Post Processing Steps
            // Read directly from native memory
            Debug.Assert(results.Count == 3, "Expecting 3 outputs");
            var boxesSpan = results[0].GetTensorDataAsSpan<float>();
            var scoresSpan = results[1].GetTensorDataAsSpan<float>();
            var indicesSpan = results[2].GetTensorDataAsSpan<int>();

            // We need multidimensional indexing to access boxes and scores. Problem is, we would have to
            // copy output data into DenseTensor via array to use it, and that can be very large. Instead, we can 
            // convert mutilidim indices into a flat index and access data straight from native memory.
            // We use DenseTensor to access multi-dimensional data
            var boxesShape = results[0].GetTensorTypeAndShape().Shape;
            var scoresShape = results[1].GetTensorTypeAndShape().Shape;
            var boxesStrides = ShapeUtils.GetStrides(boxesShape);
            var scoresStrides = ShapeUtils.GetStrides(scoresShape);

            Span<long> boxesInd = stackalloc long[boxesShape.Length];
            Span<long> scoresInd = stackalloc long[scoresShape.Length];

            var len = indicesSpan.Length / 3;
            var out_classes = new int[len];
            float[] out_scores = new float[len];
            
            var predictions = new List<Prediction>();
            var count = 0;
            for (int i = 0; i < indicesSpan.Length; i += 3)
            {
                out_classes[count] = indicesSpan[i + 1];
                if (indicesSpan[i + 1] > -1)
                {
                    // calculate scores flat index
                    scoresInd[0] = indicesSpan[i];
                    scoresInd[1] = indicesSpan[i + 1];
                    scoresInd[2] = indicesSpan[i + 2];
                    var scoresFlatIdx = ShapeUtils.GetIndex(scoresStrides, scoresInd);

                    out_scores[count] = scoresSpan[(int)scoresFlatIdx];

                    // Calculate boxes flat index
                    boxesInd[0] = indicesSpan[i];
                    boxesInd[1] = indicesSpan[i + 2];
                    boxesInd[2] = 1;
                    var idx_1 = ShapeUtils.GetIndex(boxesStrides, boxesInd);

                    boxesInd[2] = 0;
                    var idx_2 = ShapeUtils.GetIndex(boxesStrides, boxesInd);

                    boxesInd[2] = 3;
                    var idx_3 = ShapeUtils.GetIndex(boxesStrides, boxesInd);

                    boxesInd[2] = 2;
                    var idx_4 = ShapeUtils.GetIndex(boxesStrides, boxesInd);

                    predictions.Add(new Prediction
                    {
                        Box = new Box(boxesSpan[(int)idx_1],
                                      boxesSpan[(int)idx_2],
                                      boxesSpan[(int)idx_3],
                                      boxesSpan[(int)idx_4]),
                            Class = LabelMap.Labels[out_classes[count]],
                            Score = out_scores[count]
                    });
                    count++;
                }
            }

            // Put boxes, labels and confidence on image and save for viewing
            using var outputImage = File.OpenWrite(outImageFilePath);
            Font font = SystemFonts.CreateFont("Arial", 16);
            foreach (var p in predictions)
            {
                imageOrg.Mutate(x =>
                {
                    x.DrawLines(Color.Red, 2f, new PointF[] {

                        new PointF(p.Box.Xmin, p.Box.Ymin),
                        new PointF(p.Box.Xmax, p.Box.Ymin),

                        new PointF(p.Box.Xmax, p.Box.Ymin),
                        new PointF(p.Box.Xmax, p.Box.Ymax),

                        new PointF(p.Box.Xmax, p.Box.Ymax),
                        new PointF(p.Box.Xmin, p.Box.Ymax),

                        new PointF(p.Box.Xmin, p.Box.Ymax),
                        new PointF(p.Box.Xmin, p.Box.Ymin)
                    });
                    x.DrawText($"{p.Class}, {p.Score:0.00}", font, Color.White, new PointF(p.Box.Xmin, p.Box.Ymin));
                });
            }
            imageOrg.Save(outputImage, new JpegEncoder());

        }
    }
}
