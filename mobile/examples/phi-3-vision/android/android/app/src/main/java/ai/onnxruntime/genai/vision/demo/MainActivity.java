package ai.onnxruntime.genai.vision.demo;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.view.WindowManager;
import android.webkit.MimeTypeMap;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

import ai.onnxruntime.genai.GenAIException;
import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Images;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.MultiModalProcessor;
import ai.onnxruntime.genai.NamedTensors;
import ai.onnxruntime.genai.Sequences;
import ai.onnxruntime.genai.SimpleGenAI;
import ai.onnxruntime.genai.Tokenizer;
import ai.onnxruntime.genai.TokenizerStream;
import ai.onnxruntime.genai.vision.demo.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity implements Consumer<String> {

    private ActivityMainBinding binding;
    private EditText userMsgEdt;
    private Model model;
    //private Tokenizer tokenizer;
    private MultiModalProcessor multiModalProcessor;
    private ImageButton sendMsgIB;
    private ImageButton selectPhotoIB;
    private TextView generatedTV;
    private TextView promptTV;
    private TextView progressText;
    private static final String TAG = "genai.demo.MainActivity";

    private final int PICK_IMAGE_FILE = 2;
    private GenAIImage inputImage = null;

    @Override
    public void onActivityResult(int requestCode, int resultCode,
                                 Intent resultData) {
        if (requestCode == PICK_IMAGE_FILE) {
            if (resultCode == RESULT_OK) {
                // The result data contains a URI for the document or directory that
                // the user selected.
                inputImage = null;
                if (resultData != null && resultData.getData() != null) {
                    Uri uri = resultData.getData();
                    try {
                        inputImage = new GenAIImage(this, uri);
                        if (inputImage.getBitmap() != null) {
                            runOnUiThread(() -> {
                                selectPhotoIB.setImageBitmap(inputImage.getBitmap());
                            });
                        }
                    } catch (IOException | GenAIException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, resultData);
    }
    private static boolean fileExists(Context context, String fileName) {
        File file = new File(context.getFilesDir(), fileName);
        return file.exists();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        sendMsgIB = findViewById(R.id.idIBSend);
        selectPhotoIB = findViewById(R.id.idIBPhoto);
        userMsgEdt = findViewById(R.id.idEdtMessage);
        generatedTV = findViewById(R.id.sample_text);
        promptTV = findViewById(R.id.user_text);
        progressText = findViewById(R.id.idProgressStatus);

        // Trigger the download operation when the application is created
        try {
            downloadModels(
                    getApplicationContext());
        } catch (GenAIException e) {
            throw new RuntimeException(e);
        }

        Consumer<String> tokenListener = this;

        //enable scrolling and resizing of text boxes
        generatedTV.setMovementMethod(new ScrollingMovementMethod());
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);

        selectPhotoIB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                    Intent chooseFile = new Intent(Intent.ACTION_GET_CONTENT);
                    chooseFile.addCategory(Intent.CATEGORY_OPENABLE);
                    chooseFile.setType("image/*");
                    startActivityForResult(
                            Intent.createChooser(chooseFile, "Choose an image"),
                            PICK_IMAGE_FILE
                    );

            }});

        // adding on click listener for send message button.
        sendMsgIB.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (model == null) {
                    // if the edit text is empty display a toast message.
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

                String promptQuestion = "<|user|>\n";
                if (inputImage != null) {
                    promptQuestion += "<|image_1|>\n";
                }
                promptQuestion += userMsgEdt.getText().toString() + "<system>You are a helpful AI assistant. Answer in two paragraphs or less<|end|>\n<|assistant|>\n";
                final String promptQuestion_formatted = promptQuestion;

                Log.i("GenAI: prompt question", promptQuestion_formatted);
                setVisibility();

                // Disable send button while responding to prompt.
                sendMsgIB.setEnabled(false);

                promptTV.setText(userMsgEdt.getText().toString());
                // Clear Edit Text or prompt question.
                userMsgEdt.setText("");
                if (inputImage != null) {
                    generatedTV.setText("[analyzing image...]\n");
                }
                else {
                    generatedTV.setText("");
                }

                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        TokenizerStream stream = null;
                        GeneratorParams generatorParams = null;
                        Generator generator = null;
                        Sequences encodedPrompt = null;
                        Images images = null;
                        NamedTensors inputTensors = null;
                        try {
                            stream = multiModalProcessor.createStream();

                            generatorParams = model.createGeneratorParams();
                            //examples for optional parameters to format AI response
                            //generatorParams.setSearchOption("length_penalty", 1000);
                            //generatorParams.setSearchOption("max_length", 500);

                            if (inputImage != null) {
                                images = inputImage.getImages();
                            }


                            inputTensors = multiModalProcessor.processImages(promptQuestion_formatted, images);
                            generatorParams.setInput(inputTensors);

                            generator = new Generator(model, generatorParams);

                            while (!generator.isDone()) {
                                generator.computeLogits();
                                generator.generateNextToken();
                 
                                int token = generator.getLastTokenInSequence(0);
                 
                                tokenListener.accept(stream.decode(token));
                            }
                            generator.close();
                            encodedPrompt.close();
                            stream.close();
                            generatorParams.close();
                            images.close();
                            inputTensors.close();
                        }
                        catch (GenAIException e) {
                            Log.e(TAG, "Exception occurred during model query: " + e.getMessage());
                            if (generator != null) generator.close();
                            if (encodedPrompt != null) encodedPrompt.close();
                            if (stream != null) stream.close();
                            if (generatorParams != null) generatorParams.close();
                            if (images != null) images.close();
                            if (inputTensors != null) inputTensors.close();
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
        multiModalProcessor.close();
        multiModalProcessor = null;
        model.close();
        model = null;
        super.onDestroy();
    }


    private void downloadModels(Context context) throws GenAIException {

        final String baseUrl = "https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu/resolve/main/cpu-int4-rtn-block-32-acc-level-4/";
        List<String> files = Arrays.asList(
            "genai_config.json",
            "phi-3-v-128k-instruct-text-embedding.onnx",
            "phi-3-v-128k-instruct-text-embedding.onnx.data",
            "phi-3-v-128k-instruct-text.onnx",
            "phi-3-v-128k-instruct-text.onnx.data",
            "phi-3-v-128k-instruct-vision.onnx",
            "phi-3-v-128k-instruct-vision.onnx.data",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json");


        List<Pair<String, String>> urlFilePairs = new ArrayList<>();
        for (String file : files) {
            if (/*file.endsWith(".data") ||*/ !fileExists(context, file)) {
                urlFilePairs.add(new Pair<>(
                        baseUrl + file,// + "?download=true",
                        file));
            }
        }
        if (urlFilePairs.isEmpty()) {
            // Display a message using Toast
            Toast.makeText(this, "All files already exist. Skipping download.", Toast.LENGTH_SHORT).show();
            Log.d(TAG, "All files already exist. Skipping download.");
            model = new Model(getFilesDir().getPath());
            multiModalProcessor = new MultiModalProcessor(model);
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
                        model = new Model(getFilesDir().getPath());
                        multiModalProcessor = new MultiModalProcessor(model);
                        runOnUiThread(() -> {
                            Toast.makeText(context, "All downloads completed", Toast.LENGTH_SHORT).show();
                            progressText.setVisibility(View.INVISIBLE);
                        });
                    } catch (GenAIException e) {
                        e.printStackTrace();
                        throw new RuntimeException(e);
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
}
