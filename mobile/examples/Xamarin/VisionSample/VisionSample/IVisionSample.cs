// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Threading.Tasks;

namespace VisionSample
{
    public interface IVisionSample
    {
        string Name { get; }
        string ModelName { get; }
        Task InitializeAsync();
        Task UpdateExecutionProviderAsync(ExecutionProviderOptions executionProvider);
        Task<ImageProcessingResult> ProcessImageAsync(byte[] image);
    }
}
