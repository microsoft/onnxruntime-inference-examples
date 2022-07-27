// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace MauiVisionSample
{
    public enum ExecutionProviders
    {
        CPU,   // CPU execution provider is always available by default
        NNAPI, // NNAPI is available on Android
        CoreML // CoreML is available on iOS/macOS
    }
}