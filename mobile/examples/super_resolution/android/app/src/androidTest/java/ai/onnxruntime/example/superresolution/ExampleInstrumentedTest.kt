package ai.onnxruntime.example.superresolution

import ai.onnxruntime.extensions.OrtxPackage;
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {
    @Test
    fun useAppContext() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("ai.onnxruntime.example.superresolution", appContext.packageName)
    }

    @Test
    fun loadModelAndCreateOrtSession() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val resources = appContext.resources
        val modelBytes = resources.openRawResource(R.raw.pt_super_resolution_op16).readBytes()
        val env = OrtEnvironment.getEnvironment()
        env.use {
            assertNotNull(env)
            val sessionOptions: OrtSession.SessionOptions = OrtSession.SessionOptions()
            sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath())
            val session = env.createSession(modelBytes, sessionOptions)
            session.use {
                assertNotNull(session)
            }
        }
    }
}