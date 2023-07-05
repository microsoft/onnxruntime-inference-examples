// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime;

namespace image_classification
{
    class Program
    {
        static void Main(string[] args)
        {
            string modelFilePath = args[0];
            string imageRawFilePath = args[1];
            string synsetFilePath = args[2];

            using var options = new SessionOptions { LogId = "SnpeImageClassificationSample" };
            // Add SNPE EP. NB! At the time of this writing there is not a NuGet package for SNPE EP for version 1.16
            var providerOptions = new Dictionary<string, string>();
            providerOptions.Add("runtime", "DSP"); // CPU, DSP
            providerOptions.Add("buffer_type", "FLOAT");
            options.AppendExecutionProvider("SNPE", providerOptions);

            using var session = new InferenceSession(modelFilePath, options);

            // Read the image
            var input_size = 3 * 224 * 224;
            var inputRawData = new float[input_size];
            using var inputStream = new FileStream(imageRawFilePath, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(inputStream);
            for(int i = 0; i < input_size; i++)
            {
                inputRawData[i] = reader.ReadSingle();
            }

            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(inputRawData, new long[] { 1, 224, 224, 3 });
            var inputData = new Dictionary<string, OrtValue>
            {
                { "data", inputOrtValue }
            };


            // Read synset file
            var synset_size = 1000;
            var synsetData = new string[synset_size];
            string[] synset = File.ReadAllLines(synsetFilePath);

            using var runOptions = new RunOptions();
            using var results = session.Run(runOptions, inputData, session.OutputNames);
            foreach (var result in results)
            {
                var outputData = result.GetTensorDataAsSpan<float>();
                var dictionary = new Dictionary<float, int>(synset_size);
                for (int i = 0; i < synset_size; i++)
                {
                    dictionary[outputData[i]] = i;
                }
                var top5Output = outputData.ToArray().OrderByDescending(w => w).Take(5);
                foreach (var possibility in top5Output)
                {
                    Console.WriteLine("probability={0} ; class={1}", possibility, synset[dictionary[possibility]]);
                }
            }
        }
    }
}