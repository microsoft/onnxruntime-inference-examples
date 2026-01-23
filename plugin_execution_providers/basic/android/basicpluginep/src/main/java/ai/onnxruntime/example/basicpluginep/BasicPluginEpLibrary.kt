package ai.onnxruntime.example.basicpluginep

private const val basicPluginEpLibraryName = "basic_plugin_ep"
private const val basicPluginEpName = "BasicPluginExecutionProvider"

/**
 * Returns the path to the basic plugin EP library.
 * This path can be used with `OrtEnvironment.registerExecutionProviderLibrary()`.
 */
fun getLibraryPath() : String {
    return "lib${basicPluginEpLibraryName}.so"
}

/**
 * Returns the basic plugin EP name.
 * This name can be used to select an appropriate `OrtEpDevice`.
 */
fun getEpName() : String {
    return basicPluginEpName
}
