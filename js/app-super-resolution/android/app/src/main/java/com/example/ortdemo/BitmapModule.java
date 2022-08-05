package com.example.ortdemo;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.WritableNativeArray;
import com.facebook.react.bridge.WritableNativeMap;

import android.graphics.Bitmap;
import android.net.Uri;
import android.provider.MediaStore;

import androidx.annotation.NonNull;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.IntBuffer;
import java.util.Objects;

public class BitmapModule extends ReactContextBaseJavaModule {

    public BitmapModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }

    @NonNull
    @Override
    public String getName() {
        return "Bitmap";
    }

    @ReactMethod
    public void getPixels(String filePath, final Promise promise) {
        try {
            WritableNativeMap result = new WritableNativeMap();
            WritableNativeArray pixels = new WritableNativeArray();
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(Objects.requireNonNull(this.getCurrentActivity()).getContentResolver(), Uri.parse(filePath));

            if (bitmap == null) {
                promise.reject(new NullPointerException("No Bitmap Selected"));
                return;
            }

            int width = bitmap.getWidth();
            int height = bitmap.getHeight();

            boolean hasAlpha = bitmap.hasAlpha();

            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    int color = bitmap.getPixel(x, y);
                    pixels.pushInt(color);
                }
            }

            result.putInt("width", width);
            result.putInt("height", height);
            result.putBoolean("hasAlpha", hasAlpha);
            result.putArray("pixels", pixels);

            promise.resolve(result);

        } catch (Exception e) {
            promise.reject(e);
        }
    }

    @ReactMethod
    public void getImageUri(ReadableArray arrayPixels, final Promise promise) {
        try {
            WritableNativeMap result = new WritableNativeMap();
            int[] arrayIntPixels = new int[224*224*3*3];
            for (int i = 0; i < arrayPixels.size(); i++) {
                arrayIntPixels[i] = arrayPixels.getInt(i);
            }
            Bitmap bitmap = Bitmap.createBitmap(672, 672, Bitmap.Config.ARGB_8888);
            bitmap.copyPixelsFromBuffer(IntBuffer.wrap(arrayIntPixels));
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bytes);
            byte[] bitmapData = bytes.toByteArray();

            File tempFile = new File(getReactApplicationContext().getExternalFilesDir(null), "HROutput.jpg");
            FileOutputStream fos = new FileOutputStream(tempFile);
            fos.write(bitmapData);
            fos.flush();
            fos.close();
            result.putString("uri", tempFile.toString());

            promise.resolve(result);
        } catch (Exception e) {
            promise.reject(e);
        }
    }
}