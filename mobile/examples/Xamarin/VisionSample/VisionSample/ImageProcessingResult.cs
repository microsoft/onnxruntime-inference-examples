// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace VisionSample
{
    public class ImageProcessingResult
    {
        public byte[] Image { get; private set; }
        public string Caption { get; private set; }

        internal ImageProcessingResult(byte[] image, string caption = null)
        {
            Image = image;
            Caption = caption;
        }
    }
}