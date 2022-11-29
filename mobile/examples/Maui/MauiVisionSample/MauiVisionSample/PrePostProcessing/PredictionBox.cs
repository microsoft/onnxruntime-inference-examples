// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace MauiVisionSample
{
    public class PredictionBox
    {
        public float Xmin { get; set; }
        public float Ymin { get; set; }
        public float Xmax { get; set; }
        public float Ymax { get; set; }

        public PredictionBox(float xmin, float ymin, float xmax, float ymax)
        {
            Xmin = xmin;
            Ymin = ymin;
            Xmax = xmax;
            Ymax = ymax;
        }
    }
}