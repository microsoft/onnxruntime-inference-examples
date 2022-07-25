namespace MauiVisionSample
{
    internal static class Utils
    {
        internal static async Task<byte[]> LoadResource(string name)
        {
            using Stream fileStream = await FileSystem.Current.OpenAppPackageFileAsync(name);
            using MemoryStream memoryStream = new MemoryStream();
            fileStream.CopyTo(memoryStream);
            return memoryStream.ToArray();
        }
    }
}
