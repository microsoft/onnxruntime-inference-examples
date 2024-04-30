package ai.onnxruntime.genai.demo;

import android.util.Log;

public class GenAIWrapper implements AutoCloseable {
    // Load the GenAI library on application startup.
    // TODO: Do we need to load the onnxruntime library explicitly?
    // TODO: Will this work with them under
    static {
        System.loadLibrary("genai");  // JNI layer
        System.loadLibrary("onnxruntime-genai");
        System.loadLibrary("onnxruntime");
    }

    private final long nativeModel;
    private final long nativeTokenizer;
    private TokenUpdateListener listener;


    public interface TokenUpdateListener {
        void onTokenUpdate(String token);
    }

    public GenAIWrapper(String modelPath) {
        nativeModel = loadModel(modelPath);
        nativeTokenizer = createTokenizer(nativeModel);
    }

    public void setTokenUpdateListener(TokenUpdateListener listener) {
        this.listener = listener;
    }

    String run(String prompt) {
        return run(nativeModel, nativeTokenizer, prompt, /* useCallback*/ true);
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
        // TODO: Hook this up with the caller providing the callback func to the ctor of this class,
        // or alternatively to run() with it being passed into the run method
        Log.i("GenAI", "gotNextToken: " + token);
        // Call the listener method to update the token in MainActivity
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
