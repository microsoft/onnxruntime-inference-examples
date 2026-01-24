using System.Runtime.InteropServices;

namespace Contoso.ML.OnnxRuntime.EP.Basic;

public static class BasicEp
{
    private static string GetNativeDirectory()
    {
        var assemblyDir = Path.GetDirectoryName(typeof(BasicEp).Assembly.Location);

        // Try returning where this assembly lives (works for framework-dependent)
        if (!string.IsNullOrEmpty(assemblyDir) && Directory.Exists(assemblyDir))
            return assemblyDir;

        // Fallback to AppContext.BaseDirectory (works for single-file/self-contained)
        return AppContext.BaseDirectory;
    }
    private static string GetOSTag()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)) return "win";
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return "linux";
        return "unknown";
    }

    private static string GetArchTag()
    {
        return RuntimeInformation.OSArchitecture switch
        {
            Architecture.X64 => "x64",
            Architecture.Arm64 => "arm64",
            _ => "unknown"
        };
    }

    public static string GetLibraryPath()
    {
        string rootDir = GetNativeDirectory();
        string osArch = $"{GetOSTag()}-{GetArchTag()}";
        string candidatePath = Path.Combine(rootDir, "runtimes", osArch, "native", "basic_plugin_ep.dll");

        if (File.Exists(candidatePath))
        {
            return Path.GetFullPath(candidatePath);
        }

        // Not found
        return string.Empty;
    }

    public static string[] GetEpNames()
    {
        string[] ep_names = { "BasicPluginExecutionProvider" };
        return ep_names;
    }

    public static string GetEpName()
    {
        return GetEpNames()[0];
    }
}
