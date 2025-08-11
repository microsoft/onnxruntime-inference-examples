package com.example.modeltest;

import java.io.File;
import java.util.Scanner;

import ai.onnxruntime.genai.GenAIException;
import ai.onnxruntime.genai.Generator;
import ai.onnxruntime.genai.GeneratorParams;
import ai.onnxruntime.genai.Model;
import ai.onnxruntime.genai.Sequences;
import ai.onnxruntime.genai.Tokenizer;
import ai.onnxruntime.genai.TokenizerStream;

/**
 * Standalone Java application to test the Phi-3 model locally
 * 
 * Usage:
 * 1. Make sure you have the model files in the directory specified by MODEL_PATH
 * 2. Compile and run this application
 * 3. Enter prompts to test the model
 * 4. Type 'exit' to quit
 */
public class ModelTest {
    
    // Change this to the path where you have the model files locally
    private static final String MODEL_PATH = "C:\\Users\\shekadam\\.aitk\\models\\Microsoft\\Phi-3.5-mini-instruct-generic-cpu\\cpu-int4-rtn-block-32-acc-level-4";
    
    public static void main(String[] args) {
        System.out.println("Phi-3 Model Test Application");
        System.out.println("============================");
        
        // Check if model directory exists
        File modelDir = new File(MODEL_PATH);
        if (!modelDir.exists() || !modelDir.isDirectory()) {
            System.err.println("Model directory not found at: " + MODEL_PATH);
            System.err.println("Please update the MODEL_PATH in the source code.");
            return;
        }
        
        // Check for model files
        File[] onnxFiles = modelDir.listFiles((dir, name) -> name.endsWith(".onnx"));
        if (onnxFiles == null || onnxFiles.length == 0) {
            System.err.println("No ONNX files found in directory: " + MODEL_PATH);
            return;
        }
        
        System.out.println("Found " + onnxFiles.length + " ONNX files in model directory");
        System.out.println("Initializing model, please wait...");
        
        Model model = null;
        Tokenizer tokenizer = null;
        
        try {
            // Initialize model and tokenizer
            long startTime = System.currentTimeMillis();
            model = new Model(MODEL_PATH);
            tokenizer = model.createTokenizer();
            long initTime = System.currentTimeMillis() - startTime;
            
            System.out.println("Model initialized successfully in " + initTime + "ms");
            System.out.println("Enter your prompts, or type 'exit' to quit:");
            
            // Process user prompts
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.print("\nPrompt> ");
                String input = scanner.nextLine().trim();
                
                if (input.equalsIgnoreCase("exit")) {
                    break;
                }
                
                if (input.isEmpty()) {
                    continue;
                }
                
                generateResponse(model, tokenizer, input);
            }
            
            scanner.close();
            
        } catch (GenAIException e) {
            System.err.println("Error initializing model: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Clean up resources
            if (tokenizer != null) tokenizer.close();
            if (model != null) model.close();
        }
    }
    
    private static void generateResponse(Model model, Tokenizer tokenizer, String prompt) {
        TokenizerStream stream = null;
        GeneratorParams generatorParams = null;
        Sequences encodedPrompt = null;
        Generator generator = null;
        
        try {
            long startTime = System.currentTimeMillis();
            
            // Format prompt for Phi-3 model
            String promptFormatted = "<s>You are a helpful AI assistant. Answer in two paragraphs or less<|end|><|user|>" + 
                                    prompt + "<|end|>\n<assistant|>";
            
            // Create tokenizer stream
            stream = tokenizer.createStream();
            
            // Create generator parameters
            generatorParams = model.createGeneratorParams();
            generatorParams.setSearchOption("max_length", 100L);
            generatorParams.setSearchOption("temperature", 0.7);
            generatorParams.setSearchOption("top_p", 0.9);
            
            // Encode the prompt
            encodedPrompt = tokenizer.encode(promptFormatted);
            generatorParams.setInput(encodedPrompt);
            
            // Create generator
            generator = new Generator(model, generatorParams);
            
            StringBuilder result = new StringBuilder();
            System.out.print("\nGenerating: ");
            
            // Generate tokens until done
            int tokenCount = 0;
            while (!generator.isDone()) {
                generator.computeLogits();
                generator.generateNextToken();
                
                int token = generator.getLastTokenInSequence(0);
                String decodedToken = stream.decode(token);
                result.append(decodedToken);
                
                // Print progress
                System.out.print(decodedToken);
                tokenCount++;
            }
            
            long totalTime = System.currentTimeMillis() - startTime;
            double tokensPerSecond = tokenCount * 1000.0 / totalTime;
            
            System.out.println("\n\nGeneration complete!");
            System.out.println("Time: " + totalTime + "ms for " + tokenCount + " tokens (" + 
                             String.format("%.2f", tokensPerSecond) + " tokens/sec)");
            
        } catch (Exception e) {
            System.err.println("\nError generating response: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Clean up resources
            if (generator != null) generator.close();
            if (encodedPrompt != null) encodedPrompt.close();
            if (stream != null) stream.close();
            if (generatorParams != null) generatorParams.close();
        }
    }
}
