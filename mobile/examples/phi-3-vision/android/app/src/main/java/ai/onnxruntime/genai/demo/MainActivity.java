package ai.onnxruntime.genai.demo;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ai.onnxruntime.genai.demo.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity implements GenAIWrapper.TokenUpdateListener {

    private ActivityMainBinding binding;
    private EditText userMsgEdt;
    private GenAIWrapper genAIWrapper;
    private ImageButton sendMsgIB;
    private TextView generatedTV;
    private TextView promptTV;
    private static final String TAG = "genai.demo.MainActivity";

    private static boolean fileExists(Context context, String fileName) {
        File file = new File(context.getFilesDir(), fileName);
        return file.exists();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Trigger the download operation when the application is created
        try {
            downloadModels(
                    getApplicationContext());
        } catch (GenAIException e) {
            throw new RuntimeException(e);
        }

        sendMsgIB = findViewById(R.id.idIBSend);
        userMsgEdt = findViewById(R.id.idEdtMessage);
        generatedTV = findViewById(R.id.sample_text);
        promptTV = findViewById(R.id.user_text);

        // adding on click listener for send message button.
        sendMsgIB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Checking if the message entered
                // by user is empty or not.
                if (userMsgEdt.getText().toString().isEmpty()) {
                    // if the edit text is empty display a toast message.
                    Toast.makeText(MainActivity.this, "Please enter your message..", Toast.LENGTH_SHORT).show();
                    return;
                }

                String promptQuestion = userMsgEdt.getText().toString();
                String promptQuestion_formatted = "<|user|>\n" + promptQuestion + "<|end|>\n<|assistant|>";
                Log.i("GenAI: prompt question", promptQuestion_formatted);
                setVisibility();

                // Disable send button while responding to prompt.
                sendMsgIB.setEnabled(false);

                promptTV.setText(promptQuestion);
                // Clear Edit Text or prompt question.
                userMsgEdt.setText("");
                generatedTV.setText("");

                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            genAIWrapper.run(promptQuestion_formatted);
                        } catch (GenAIException e) {
                            throw new RuntimeException(e);
                        }

                        runOnUiThread(() -> {
                            sendMsgIB.setEnabled(true);
                        });
                    }
                }).start();
            }
        });
    }

    @Override
    protected void onDestroy() {
        try {
            genAIWrapper.close();
        } catch (Exception e) {
            Log.e(TAG, "exception from closing genAIWrapper", e);
        }
        genAIWrapper = null;
        super.onDestroy();
    }

    private void downloadModels(Context context) throws GenAIException {
        List<Pair<String, String>> urlFilePairs = Arrays.asList(
                new Pair<>(
                        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/added_tokens.json?download=true",
                        "added_tokens.json"),
                new Pair<>(
                        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/config.json?download=true",
                        "config.json"),
                new Pair<>(
                        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/configuration_phi3.py?download=true",
                        "configuration_phi3.py"),
                new Pair<>(
                        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/genai_config.json?download=true",
                        "genai_config.json"),
                new Pair<>(
                        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx?download=true",
                        "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx"),
                new Pair<>(
                        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data?download=true",
                        "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data"),
                new Pair<>(
                        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/special_tokens_map.json?download=true",
                        "special_tokens_map.json"),
                new Pair<>(
                        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer.json?download=true",
                        "tokenizer.json"),
                new Pair<>(
                        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer.model?download=true",
                        "tokenizer.model"),
                new Pair<>(
                        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/tokenizer_config.json?download=true",
                        "tokenizer_config.json"));
        Toast.makeText(this,
                "Downloading model for the app... Model Size greater than 2GB, please allow a few minutes to download.",
                Toast.LENGTH_SHORT).show();

        ExecutorService executor = Executors.newSingleThreadExecutor();
        for (int i = 0; i < urlFilePairs.size(); i++) {
            final int index = i;
            String url = urlFilePairs.get(index).first;
            String fileName = urlFilePairs.get(index).second;
            if (fileExists(context, fileName)) {
                // Display a message using Toast
                Toast.makeText(this, "File already exists. Skipping Download.", Toast.LENGTH_SHORT).show();

                Log.d(TAG, "File " + fileName + " already exists. Skipping download.");
                // note: since we always download the files lists together for once,
                // so assuming if one filename exists, then the download model step has already
                // be
                // done.
                genAIWrapper = createGenAIWrapper();
                break;
            }
            executor.execute(() -> {
                ModelDownloader.downloadModel(context, url, fileName, new ModelDownloader.DownloadCallback() {
                    @Override
                    public void onDownloadComplete() throws GenAIException {
                        Log.d(TAG, "Download complete for " + fileName);
                        if (index == urlFilePairs.size() - 1) {
                            // Last download completed, create GenAIWrapper
                            genAIWrapper = createGenAIWrapper();
                            Log.d(TAG, "All downloads completed");
                        }
                    }
                });
            });
        }
        executor.shutdown();
    }

    private GenAIWrapper createGenAIWrapper() throws GenAIException {
        // Create GenAIWrapper object and load model from android device file path.
        GenAIWrapper wrapper = new GenAIWrapper(getFilesDir().getPath());
        wrapper.setTokenUpdateListener(this);
        return wrapper;
    }

    @Override
    public void onTokenUpdate(String token) {
        runOnUiThread(() -> {
            // Update and aggregate the generated text and write to text box.
            CharSequence generated = generatedTV.getText();
            generatedTV.setText(generated + token);
            generatedTV.invalidate();
        });
    }

    public void setVisibility() {
        TextView view = (TextView) findViewById(R.id.user_text);
        view.setVisibility(View.VISIBLE);
        TextView botView = (TextView) findViewById(R.id.sample_text);
        botView.setVisibility(View.VISIBLE);
    }
}
