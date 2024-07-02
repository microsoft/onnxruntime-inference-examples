package ai.onnxruntime.genai.demo;

import static androidx.constraintlayout.helper.widget.MotionEffect.TAG;

import android.content.Context;
import android.util.Log;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class ModelDownloader {
  interface DownloadCallback {
    void onDownloadComplete() throws GenAIException;
  }

  public static void downloadModel(Context context, String url, String fileName, DownloadCallback callback) {
    try {
      File file = new File(context.getFilesDir(), fileName);
      File tempFile = new File(context.getFilesDir(), fileName + ".tmp");
      URL modelUrl = new URL(url);
      HttpURLConnection connection = (HttpURLConnection) modelUrl.openConnection();
      connection.connect();

      // Check if response code is OK
      if (connection.getResponseCode() == HttpURLConnection.HTTP_OK) {
        InputStream inputStream = connection.getInputStream();
        FileOutputStream outputStream = new FileOutputStream(tempFile);

        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = inputStream.read(buffer)) != -1) {
          outputStream.write(buffer, 0, bytesRead);
        }

        outputStream.flush();
        outputStream.close();
        inputStream.close();

        // File downloaded successfully
        if (tempFile.renameTo(file)) {
          Log.d(TAG, "File downloaded successfully");
          if (callback != null) {
            callback.onDownloadComplete();
          }
        } else {
          Log.e(TAG, "Failed to rename temp file to original file");
        }
      } else {
        Log.e(TAG, "Failed to download model. HTTP response code: " + connection.getResponseCode());
      }
    } catch (IOException e) {
      e.printStackTrace();
      Log.e(TAG, "Exception occurred during model download: " + e.getMessage());
    } catch (GenAIException e) {
        throw new RuntimeException(e);
    }
  }
}