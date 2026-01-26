using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Contoso.ML.OnnxRuntime.EP.Basic;

public static class BasicEp
{
    /// <summary>
    /// Returns the path to the plugin EP library DLL contained by this package.
    /// Can be passed to OrtEnv::RegisterExecutionProviderLibrary().
    ///
    /// Note: It is recommended that plugin EP packages provide this information to applications.
    /// </summary>
    /// <returns>EP library path</returns>
    /// <exception cref="FileNotFoundException">If the EP DLL file path does not exist</exception>
    public static string GetLibraryPath()
    {
        string rootDir = GetNativeDirectory();
        string osArch = $"{GetOSTag()}-{GetArchTag()}";
        string epDllPath = Path.GetFullPath(Path.Combine(rootDir, "runtimes", osArch,
                                                         "native", "basic_plugin_ep.dll"));

        if (!File.Exists(epDllPath))
        {
            // This indicates a packaging error.
            throw new FileNotFoundException($"Did not find EP DLL file: {epDllPath}");
        }

        return epDllPath;
    }

    /// <summary>
    /// Returns the names of the EPs created by the plugin EP library.
    /// Can be used to select a OrtEpDevice from those returned by OrtEnv::GetEpDevices().
    ///
    /// Note: It is recommended that plugin EP packages provide this information to applications.
    /// </summary>
    /// <returns>Array of EP names</returns>
    public static string[] GetEpNames()
    {
        return ["BasicPluginExecutionProvider"];
    }

    /// <summary>
    /// Returns the name of the one EP supported by this plugin EP library.
    ///
    /// Note: This is a convenience function exposed by plugin EP packages that only have one EP name.
    /// </summary>
    /// <returns></returns>
    public static string GetEpName()
    {
        return GetEpNames()[0];
    }

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
}
