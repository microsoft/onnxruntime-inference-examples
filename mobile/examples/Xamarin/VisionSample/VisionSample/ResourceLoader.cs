// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.IO;
using System.Linq;
using System.Reflection;

namespace VisionSample
{
    public static class ResourceLoader
    {
        public static byte[] GetEmbeddedResource(string name, Assembly assembly = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException($"Parameter {nameof(name)} cannot be null or whitespace");

            if (assembly == null) assembly = typeof(ResourceLoader).Assembly;

            var resourceName = assembly.GetManifestResourceNames().FirstOrDefault(i => i.EndsWith(name));

            if (string.IsNullOrWhiteSpace(resourceName))
                throw new Exception($"Unable to resolve an embedded resource named {name}");

            using Stream stream = assembly.GetManifestResourceStream(resourceName);
            using MemoryStream memoryStream = new MemoryStream();

            stream.CopyTo(memoryStream);

            return memoryStream.ToArray();
        }

        public static bool EmbeddedResourceExists(string name, Assembly assembly = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException($"Parameter {nameof(name)} cannot be null or whitespace");

            if (assembly == null) assembly = typeof(ResourceLoader).Assembly;

            var resourceName = assembly.GetManifestResourceNames().FirstOrDefault(i => i.EndsWith(name));

            return !string.IsNullOrWhiteSpace(resourceName);
        }
    }
}