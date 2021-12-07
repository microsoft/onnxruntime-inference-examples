// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;

namespace VisionSample
{
    public class VisionSampleBase<TImageProcessor> : IVisionSample where TImageProcessor : new()
    {
        byte[] _model;
        string _name;
        string _modelName;
        Task _prevAsyncTask;
        TImageProcessor _imageProcessor;
        InferenceSession _session;
        ExecutionProviderOptions _curExecutionProvider;

        public VisionSampleBase(string name, string modelName)
        {
            _name = name;
            _modelName = modelName;
            _ = InitializeAsync();
        }

        public string Name => _name;
        public string ModelName => _modelName;
        public byte[] Model => _model;
        public InferenceSession Session => _session;
        public TImageProcessor ImageProcessor => _imageProcessor ??= new TImageProcessor();

        public async Task UpdateExecutionProviderAsync(ExecutionProviderOptions executionProvider)
        {
            // make sure any existing async task completes before we change the session
            await AwaitLastTaskAsync();

            // creating the inference session can be expensive and should be done as a one-off.
            // additionally each session uses memory for the model and the infrastructure required to execute it,
            // and has its own threadpools.
            _prevAsyncTask = Task.Run(() =>
                            {
                                if (executionProvider == _curExecutionProvider)
                                    return;

                                if (executionProvider == ExecutionProviderOptions.CPU)
                                {
                                    // create session that uses the CPU execution provider
                                    _session = new InferenceSession(_model);
                                }
                                else
                                {
                                    // create session that uses the NNAPI/CoreML. the CPU execution provider is also
                                    // enabled by default to handle any parts of the model that NNAPI/CoreML cannot.
                                    var options = SessionOptionsContainer.Create(nameof(ExecutionProviderOptions.Platform));
                                    _session = new InferenceSession(_model, options);
                                }
                            });
        }

        protected virtual Task<ImageProcessingResult> OnProcessImageAsync(byte[] image) => throw new NotImplementedException();

        public Task InitializeAsync()
        {
            _prevAsyncTask = Task.Run(() => Initialize());
            return _prevAsyncTask;
        }

        public async Task<ImageProcessingResult> ProcessImageAsync(byte[] image)
        {
            await AwaitLastTaskAsync().ConfigureAwait(false);

            return await OnProcessImageAsync(image);
        }

        async Task AwaitLastTaskAsync()
        {
            if (_prevAsyncTask != null)
            {
                await _prevAsyncTask.ConfigureAwait(false);
                _prevAsyncTask = null;
            }
        }

        void Initialize()
        {
            _model = ResourceLoader.GetEmbeddedResource(ModelName);
            _session = new InferenceSession(_model);  // default to CPU 
            _curExecutionProvider = ExecutionProviderOptions.CPU;
        }
    }
}
