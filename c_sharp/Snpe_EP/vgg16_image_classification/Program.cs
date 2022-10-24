// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Xml.Linq;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace image_classification
{
    class Program
    {
        static void Main(string[] args)
        {
            string modelFilePath = args[0];
            string imageRawFilePath = args[1];
            string synsetFilePath = args[2];

            // Read the image
            var input_size = 3 * 224 * 224;
            var inputRawData = new float[input_size];
            var inputStream = new FileStream(imageRawFilePath, FileMode.Open, FileAccess.Read);
            var reader = new BinaryReader(inputStream);
            for(int i = 0; i < input_size; i++)
            {
                inputRawData[i] = reader.ReadSingle();
            }
            var inputData = new List<NamedOnnxValue>();
            var tensor = new DenseTensor<float>(inputRawData, new int[] { 1, 224, 224, 3 });
            inputData.Add(NamedOnnxValue.CreateFromTensor<float>("data", tensor));

            // Read synset file
            var synset_size = 1000;
            var synsetData = new string[synset_size];
            string[] synset = File.ReadAllLines(synsetFilePath);


            var options = new SessionOptions { LogId = "SnpeImageClassificationSample" };

            // Add SNPE EP
            var providerOptions = new Dictionary<string, string>();
            providerOptions.Add("runtime", "DSP"); // CPU, DSP
            providerOptions.Add("buffer_type", "FLOAT");
            options.AppendExecutionProvider("SNPE", providerOptions);

            using var session = new InferenceSession(modelFilePath, options);

            foreach (var result in session.Run(inputData))
            {
                var outputData = result.AsTensor<float>().ToArray();
                var dictionary = new Dictionary<float, int>(synset_size);
                for (int i = 0; i < synset_size; i++)
                {
                    dictionary[outputData[i]] = i;
                }
                var top5Output = outputData.OrderByDescending(w => w).Take(5);
                foreach (var possibility in top5Output)
                {
                    Console.WriteLine("probability={0} ; class={1}", possibility, synset[dictionary[possibility]]);
                }
            }
        }
    }
}