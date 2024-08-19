package ai.onnxruntime.genai.vision.demo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

import ai.onnxruntime.genai.GenAIException;
import ai.onnxruntime.genai.Images;

public class GenAIImage {
    Images images = null;
    Bitmap bitmap = null;

    GenAIImage(Context context, Uri uri, final int maxWidth, final int maxHeight) throws IOException, GenAIException {
        Bitmap bmp = decodeUri(context, uri, maxWidth, maxHeight);
        String filename = context.getFilesDir() + "/multimodalinput.png";
        FileOutputStream out = new FileOutputStream(filename);
        bmp.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
        // PNG is a lossless format, the compression factor (100) is ignored
        images = new Images(filename);
        images = new Images(filename);
        bitmap = BitmapFactory.decodeFile(filename);
    }

    GenAIImage(Context context, Uri uri) throws IOException, GenAIException {
        this(context, uri, 100000, 100000);
    }

    public Images getImages() {
        return images;
    }

    public Bitmap getBitmap() { return bitmap; }

    private static Bitmap decodeUri(Context c, Uri uri, final int maxWidth, final int maxHeight)
            throws FileNotFoundException {
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(c.getContentResolver().openInputStream(uri), null, o);

        int width_tmp = o.outWidth
                , height_tmp = o.outHeight;
        int scale = 1;

        while(width_tmp / 2 > maxWidth || height_tmp / 2 > maxHeight) {
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        return BitmapFactory.decodeStream(c.getContentResolver().openInputStream(uri), null, o2);
    }
}
