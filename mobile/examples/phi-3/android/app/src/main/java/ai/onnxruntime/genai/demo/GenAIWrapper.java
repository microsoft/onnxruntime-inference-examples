package ai.onnxruntime.genai.demo;

import android.util.Log;

public class GenAIWrapper implements AutoCloseable {
    // Load the GenAI library on application startup.
    static {
        System.loadLibrary("genai"); // JNI layer
        System.loadLibrary("onnxruntime-genai");
        System.loadLibrary("onnxruntime");
    }

    private final long nativeModel;
    private final long nativeTokenizer;
    private TokenUpdateListener listener;

    public interface TokenUpdateListener {
        void onTokenUpdate(String token);
    }

    public GenAIWrapper(String modelPath) throws GenAIException {
        nativeModel = loadModel(modelPath);
        nativeTokenizer = createTokenizer(nativeModel);
    }

    public void setTokenUpdateListener(TokenUpdateListener listener) {
        this.listener = listener;
    }

    String run(String prompt) {
        return run(nativeModel, nativeTokenizer, prompt, /* useCallback */ true);
    }

    @Override
    public void close() throws Exception {
        if (nativeTokenizer != 0) {
            releaseTokenizer(nativeTokenizer);
        }

        if (nativeModel != 0) {
            releaseModel(nativeModel);
        }
    }

    public void gotNextToken(String token) {
        Log.i("GenAI", "gotNextToken: " + token);
        if (listener != null) {
            listener.onTokenUpdate(token);
        }
    }

    private native long loadModel(String modelPath);

    private native void releaseModel(long nativeModel);

    private native long createTokenizer(long nativeModel);

    private native void releaseTokenizer(long nativeTokenizer);

    private native String run(long nativeModel, long nativeTokenizer, String prompt, boolean useCallback);
}
