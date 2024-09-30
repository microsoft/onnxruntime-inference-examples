package ai.onnxruntime.genai.demo;

import static androidx.constraintlayout.helper.widget.MotionEffect.TAG;

import android.content.Context;
import android.util.Log;
import android.util.Pair;
import android.widget.Toast;

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
import java.util.ArrayList;
import java.util.List;

import ai.onnxruntime.genai.GenAIException;

public class ModelDownloader {
  interface DownloadCallback {
    void onProgress(long lastBytesRead, long bytesRead, long bytesTotal);
    void onDownloadComplete() throws GenAIException;
  }

  public static void downloadModel(Context context, List<Pair<String, String>> urlFilePairs, DownloadCallback callback) {
    try {

      List<HttpURLConnection> connections = new ArrayList<>();
      long totalDownloadBytes = 0;
      for (int i = 0; i < urlFilePairs.size(); i++) {
        String url = urlFilePairs.get(i).first;
        URL modelUrl = new URL(url);
        HttpURLConnection connection = (HttpURLConnection) modelUrl.openConnection();
        connections.add(connection);
        long totalFileSize = connection.getHeaderFieldLong("Content-Length",-1);
        totalDownloadBytes += totalFileSize;
      }

      long totalBytesRead = 0;
      for (int i = 0; i < urlFilePairs.size(); i++) {
        String fileName = urlFilePairs.get(i).second;
        HttpURLConnection connection = connections.get(i);

        File file = new File(context.getFilesDir(), fileName);
        File tempFile = new File(context.getFilesDir(), fileName + ".tmp");
        Log.d(TAG, "Downloading file: " + fileName);
        connection.connect();

        // Check if response code is OK
        if (connection.getResponseCode() == HttpURLConnection.HTTP_OK) {
          InputStream inputStream = connection.getInputStream();
          FileOutputStream outputStream = new FileOutputStream(tempFile);

          long begin = System.currentTimeMillis();

          byte[] buffer = new byte[4096];
          int bytesRead;

          while ((bytesRead = inputStream.read(buffer)) != -1) {
            outputStream.write(buffer, 0, bytesRead);
            if (callback != null) {
              callback.onProgress(totalBytesRead, totalBytesRead + bytesRead, totalDownloadBytes);
            }
            totalBytesRead += bytesRead;
          }

          outputStream.flush();
          outputStream.close();
          inputStream.close();
          connection.disconnect();

          long duration = System.currentTimeMillis() - begin;

          // File downloaded successfully
          if (tempFile.renameTo(file)) {
            if (duration > 0) {
              Log.d(TAG, "File downloaded successfully: " + fileName + "(" + totalBytesRead + " bytes, " + (totalBytesRead / duration) + "KBps)");
            } else {
              Log.d(TAG, "File downloaded successfully: " + fileName + "(" + totalBytesRead + " bytes, " + (duration / 1000.0) + "s)");
            }
          } else {
            Log.e(TAG, "Failed to rename temp file to original file");
          }
        } else {
          Log.e(TAG, "Failed to download model. HTTP response code: " + connection.getResponseCode());
        }
      }
      if (callback != null) {
        callback.onDownloadComplete();
      }
    } catch (IOException e) {
      e.printStackTrace();
      Log.e(TAG, "Exception occurred during model download: " + e.getMessage());
    } catch (GenAIException e) {
      throw new RuntimeException(e);
    }
  }
}