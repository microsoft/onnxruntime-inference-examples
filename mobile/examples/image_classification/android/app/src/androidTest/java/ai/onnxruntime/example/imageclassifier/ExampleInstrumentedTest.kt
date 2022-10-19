package ai.onnxruntime.example.imageclassifier

import ai.onnxruntime.OrtEnvironment
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith

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
        assertEquals("ai.onnxruntime.example.imageclassifier", appContext.packageName)
    }

    @Test
    fun loadModel() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val resources = appContext.getResources()
        val modelBytes = resources.openRawResource(R.raw.mobilenetv2_fp32).readBytes()
        val env = OrtEnvironment.getEnvironment()
        env.use {
            assertNotNull(env)
            val session = env.createSession(modelBytes)
            session.use {
                assertNotNull(session)
            }
        }
    }
}
