// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;

namespace MauiVisionSample
{
    public class VisionSampleBase<TImageProcessor> : IVisionSample where TImageProcessor : new()
    {
        byte[] _model;
        string _name;
        string _modelName;
        Task _prevAsyncTask;
        TImageProcessor _imageProcessor;
        InferenceSession _session;
        ExecutionProviders _curExecutionProvider;

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

        public async Task UpdateExecutionProviderAsync(ExecutionProviders executionProvider)
        {
            // make sure any existing async task completes before we change the session
            await AwaitLastTaskAsync();

            // creating the inference session can be expensive and should be done as a one-off.
            // additionally each session uses memory for the model and the infrastructure required to execute it,
            // and has its own threadpools.
            _prevAsyncTask = 
                Task.Run(() =>
                {
                    if (executionProvider == _curExecutionProvider)
                    { 
                        return; 
                    }

                    var options = new SessionOptions();

                    if (executionProvider == ExecutionProviders.CPU)
                    {
                        // CPU Execution Provider is always enabled
                    }
                    else if (executionProvider == ExecutionProviders.NNAPI)
                    {
                        options.AppendExecutionProvider_Nnapi();
                    }
                    else if (executionProvider == ExecutionProviders.CoreML)
                    {
                        // add CoreML if the device has an Apple Neural Engine. if it doesn't performance
                        // will most likely be worse than with the CPU Execution Provider.
                        options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
                    }

                    _session = new InferenceSession(_model, options);
                });
        }

        protected virtual Task<ImageProcessingResult> OnProcessImageAsync(byte[] image) => 
            throw new NotImplementedException();

        public Task InitializeAsync()
        {
            _prevAsyncTask = Initialize();
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

        async Task Initialize()
        {
            _model = await Utils.LoadResource(_modelName);
            _session = new InferenceSession(_model);  // CPU execution provider is always enabled
            _curExecutionProvider = ExecutionProviders.CPU;
        }
    }
}
