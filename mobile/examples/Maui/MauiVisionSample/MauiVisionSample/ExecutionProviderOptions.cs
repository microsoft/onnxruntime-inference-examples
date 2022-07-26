// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace MauiVisionSample
{
    public enum ExecutionProviderOptions
    {
        CPU,      // default CPU execution provider
        Platform  // platform specific provider. NNAPI on Android, CoreML on iOS. 
    }
}