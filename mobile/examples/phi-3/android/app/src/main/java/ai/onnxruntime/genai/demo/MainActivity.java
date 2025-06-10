package ai.onnxruntime.genai.demo;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Dialog;
import android.content.Context;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
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
import java.util.function.Consumer;

import ai.onnxruntime.genai.SimpleGenAI;
import ai.onnxruntime.genai.GenAIException;
import ai.onnxruntime.genai.GeneratorParams;

public class MainActivity extends AppCompatActivity implements Consumer<String> {

    // ===== MODEL CONFIGURATION - MODIFY THESE FOR DIFFERENT MODELS =====
    // Base URL for downloading model files (ensure it ends with '/')
    private static final String MODEL_BASE_URL = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/resolve/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/";
    
    // List of required model files to download
    private static final List<String> MODEL_FILES = Arrays.asList(
            "added_tokens.json",
            "config.json",
            "configuration_phi3.py", 
            "genai_config.json",
            "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx",
            "phi3-mini-4k-instruct-cpu-int4-rtn-block-32-acc-level-4.onnx.data",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json"
    );
    // ===== END MODEL CONFIGURATION =====

    private EditText userMsgEdt;
    private SimpleGenAI genAI;
    private ImageButton sendMsgIB;
    private TextView generatedTV;
    private TextView promptTV;
    private TextView progressText;
    private ImageButton settingsButton;
    private static final String TAG = "genai.demo.MainActivity";
    private int maxLength = 100;
    private float lengthPenalty = 1.0f;

    private static boolean fileExists(Context context, String fileName) {
        File file = new File(context.getFilesDir(), fileName);
        return file.exists();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        sendMsgIB = findViewById(R.id.idIBSend);
        userMsgEdt = findViewById(R.id.idEdtMessage);
        generatedTV = findViewById(R.id.sample_text);
        promptTV = findViewById(R.id.user_text);
        progressText = findViewById(R.id.progress_text);
        settingsButton = findViewById(R.id.idIBSettings);

        // Trigger the download operation when the application is created
        try {
            downloadModels(
                    getApplicationContext());
        } catch (GenAIException e) {
            throw new RuntimeException(e);
        }

        settingsButton.setOnClickListener(v -> {
            BottomSheet bottomSheet = new BottomSheet();
            bottomSheet.setSettingsListener(new BottomSheet.SettingsListener() {
                @Override
                public void onSettingsApplied(int maxLength, float lengthPenalty) {
                    MainActivity.this.maxLength = maxLength;
                    MainActivity.this.lengthPenalty = lengthPenalty;
                    Log.i(TAG, "Setting max response length to: " + maxLength);
                    Log.i(TAG, "Setting length penalty to: " + lengthPenalty);
                }
            });
            bottomSheet.show(getSupportFragmentManager(), "BottomSheet");
        });


        //enable scrolling and resizing of text boxes
        generatedTV.setMovementMethod(new ScrollingMovementMethod());
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        // adding on click listener for send message button.
        sendMsgIB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (genAI == null) {
                    // if user tries to submit prompt while model is still downloading, display a toast message.
                    Toast.makeText(MainActivity.this, "Model not loaded yet, please wait...", Toast.LENGTH_SHORT).show();
                    return;
                }

                // Checking if the message entered
                // by user is empty or not.
                if (userMsgEdt.getText().toString().isEmpty()) {
                    // if the edit text is empty display a toast message.
                    Toast.makeText(MainActivity.this, "Please enter your message..", Toast.LENGTH_SHORT).show();
                    return;
                }

                String promptQuestion = userMsgEdt.getText().toString();
                String promptQuestion_formatted = "<system>You are a helpful AI assistant. Answer in two paragraphs or less<|end|><|user|>"+promptQuestion+"<|end|>\n<assistant|>";
                Log.i("GenAI: prompt question", promptQuestion_formatted);
                setVisibility();

                // Disable send button while responding to prompt.
                sendMsgIB.setEnabled(false);
                sendMsgIB.setAlpha(0.5f);

                promptTV.setText(promptQuestion);
                // Clear Edit Text or prompt question.
                userMsgEdt.setText("");
                generatedTV.setText("");

                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            // Create generator parameters
                            GeneratorParams generatorParams = genAI.createGeneratorParams();
                            
                            // Set optional parameters to format AI response
                            // https://onnxruntime.ai/docs/genai/reference/config.html
                            generatorParams.setSearchOption("length_penalty", (double)lengthPenalty);
                            generatorParams.setSearchOption("max_length", (double)maxLength);
                            long startTime = System.currentTimeMillis();
                            final long[] firstTokenTime = {startTime};
                            final long[] numTokens = {0};
                            
                            // Token listener for streaming tokens
                            Consumer<String> tokenListener = token -> {
                                if (numTokens[0] == 0) {
                                    firstTokenTime[0] = System.currentTimeMillis();
                                }

                                
                                // Update UI with new token
                                MainActivity.this.accept(token);
                                
                                Log.i(TAG, "Generated token: " + token);
                                numTokens[0] += 1;
                            };

                            String fullResponse = genAI.generate(generatorParams, promptQuestion_formatted, tokenListener);
                            
                            long totalTime = System.currentTimeMillis() - firstTokenTime[0];
                            float promptProcessingTime = (firstTokenTime[0] - startTime) / 1000.0f;
                            float tokensPerSecond = numTokens[0] > 1 ? (1000.0f * (numTokens[0] - 1)) / totalTime : 0;

                            runOnUiThread(() -> {
                                showTokenPopup(promptProcessingTime, tokensPerSecond);
                            });

                            Log.i(TAG, "Full response: " + fullResponse);
                            Log.i(TAG, "Prompt processing time (first token): " + promptProcessingTime + " seconds");
                            Log.i(TAG, "Tokens generated per second (excluding prompt processing): " + tokensPerSecond);
                        }
                        catch (GenAIException e) {
                            Log.e(TAG, "Exception occurred during model query: " + e.getMessage());
                            runOnUiThread(() -> {
                                Toast.makeText(MainActivity.this, "Error generating response: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                            });
                        }
                        finally {
                            runOnUiThread(() -> {
                                sendMsgIB.setEnabled(true);
                                sendMsgIB.setAlpha(1.0f);
                            });
                        }
                    }
                }).start();
            }
        });
    }

    @Override
    protected void onDestroy() {
        if (genAI != null) {
            genAI.close();
            genAI = null;
        }
        super.onDestroy();
    }

    private void downloadModels(Context context) throws GenAIException {

        List<Pair<String, String>> urlFilePairs = new ArrayList<>();
        for (String file : MODEL_FILES) {
            if (!fileExists(context, file)) {
                urlFilePairs.add(new Pair<>(
                        MODEL_BASE_URL + file,
                        file));
            }
        }
        if (urlFilePairs.isEmpty()) {
            // Display a message using Toast
            Toast.makeText(this, "All files already exist. Skipping download.", Toast.LENGTH_SHORT).show();
            Log.d(TAG, "All files already exist. Skipping download.");
            genAI = new SimpleGenAI(getFilesDir().getPath());
            return;
        }

        progressText.setText("Downloading...");
        progressText.setVisibility(View.VISIBLE);

        Toast.makeText(this,
                "Downloading model for the app... Model Size greater than 2GB, please allow a few minutes to download.",
                Toast.LENGTH_SHORT).show();

        ExecutorService executor = Executors.newSingleThreadExecutor();
        executor.execute(() -> {
            ModelDownloader.downloadModel(context, urlFilePairs, new ModelDownloader.DownloadCallback() {
                @Override
                public void onProgress(long lastBytesRead, long bytesRead, long bytesTotal) {
                    long lastPctDone = 100 * lastBytesRead / bytesTotal;
                    long pctDone = 100 * bytesRead / bytesTotal;
                    if (pctDone > lastPctDone) {
                        Log.d(TAG, "Downloading files: " + pctDone + "%");
                        runOnUiThread(() -> {
                            progressText.setText("Downloading: " + pctDone + "%");
                        });
                    }
                }
                @Override
                public void onDownloadComplete() {
                    Log.d(TAG, "All downloads completed.");

                    // Last download completed, create SimpleGenAI
                    try {
                        genAI = new SimpleGenAI(getFilesDir().getPath());
                        runOnUiThread(() -> {
                            Toast.makeText(context, "All downloads completed", Toast.LENGTH_SHORT).show();
                            progressText.setVisibility(View.INVISIBLE);
                        });
                    } catch (GenAIException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to initialize SimpleGenAI: " + e.getMessage());
                        runOnUiThread(() -> {
                            Toast.makeText(context, "Failed to load model: " + e.getMessage(), Toast.LENGTH_LONG).show();
                            progressText.setText("Failed to load model");
                        });
                    }

                }
            });
        });
        executor.shutdown();
    }

    @Override
    public void accept(String token) {
        runOnUiThread(() -> {
            // Update and aggregate the generated text and write to text box.
            CharSequence generated = generatedTV.getText();
            generatedTV.setText(generated + token);
            generatedTV.invalidate();
            final int scrollAmount = generatedTV.getLayout().getLineTop(generatedTV.getLineCount()) - generatedTV.getHeight();
            generatedTV.scrollTo(0, Math.max(scrollAmount, 0));
        });
    }

    public void setVisibility() {
        TextView view = (TextView) findViewById(R.id.user_text);
        view.setVisibility(View.VISIBLE);
        TextView botView = (TextView) findViewById(R.id.sample_text);
        botView.setVisibility(View.VISIBLE);
    }

    private void showTokenPopup(float promptProcessingTime, float tokenRate) {

        final Dialog dialog = new Dialog(MainActivity.this);
        dialog.setContentView(R.layout.info_popup);

        TextView promptProcessingTimeTv = dialog.findViewById(R.id.prompt_processing_time_tv);
        TextView tokensPerSecondTv = dialog.findViewById(R.id.tokens_per_second_tv);
        Button closeBtn = dialog.findViewById(R.id.close_btn);

        promptProcessingTimeTv.setText(String.format("Prompt processing time: %.2f seconds", promptProcessingTime));
        tokensPerSecondTv.setText(String.format("Tokens per second: %.2f", tokenRate));

        closeBtn.setOnClickListener(v -> dialog.dismiss());

        dialog.show();
    }


}
