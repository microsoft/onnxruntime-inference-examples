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
 * Returns the name of the EP provided by the basic plugin EP library.
 * This name can be used to select an appropriate `OrtEpDevice`.
 */
fun getEpName() : String {
    return basicPluginEpName
}

/**
 * Returns the names of the EPs provided by the basic plugin EP library. There is only one.
 * These names can be used to select an appropriate `OrtEpDevice`.
 */
fun getEpNames() : Array<String> {
    return arrayOf(getEpName())
}
