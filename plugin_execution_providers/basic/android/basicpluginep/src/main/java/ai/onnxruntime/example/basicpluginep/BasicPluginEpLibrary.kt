package ai.onnxruntime.example.basicpluginep

private const val basicPluginEpLibraryName = "basic_plugin_ep"

/**
 * Returns the path to the basic plugin EP library.
 * This path can be used with `OrtEnvironment.registerExecutionProviderLibrary()`.
 */
fun getBasicPluginEpLibraryPath() : String {
    return "lib${basicPluginEpLibraryName}.so"
}