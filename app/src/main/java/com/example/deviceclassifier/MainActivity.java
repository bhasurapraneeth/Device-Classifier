package com.example.deviceclassifier;


import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    private Uri imageUri;
    private ImageView imageView;
    private Interpreter tfliteInterpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        Button selectImageButton = findViewById(R.id.selectImageButton);
        Button predictButton = findViewById(R.id.predictButton);
        TextView predictionTextView = findViewById(R.id.predictionTextView);

        // Load the TensorFlow Lite model
        try {
            tfliteInterpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Error loading TFLite model!", Toast.LENGTH_SHORT).show();
        }

        // Button to select an image from the gallery
        selectImageButton.setOnClickListener(v -> {
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_GET_CONTENT);
            startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_REQUEST);
        });

        // Button to run inference on the selected image
        predictButton.setOnClickListener(v -> {
            if (imageUri != null) {
                runInference(predictionTextView);
            } else {
                Toast.makeText(this, "Please select an image first!", Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            imageUri = data.getData();
            imageView.setImageURI(imageUri);  // Display the selected image in the ImageView
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        // Load the TFLite model from the assets folder
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("device_classifier_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void runInference(TextView predictionTextView) {
        try {
            // Load and preprocess the selected image
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);  // Resize to 224x224

            // Convert the image to a float array
            float[][][][] input = new float[1][224][224][3];
            for (int y = 0; y < 224; y++) {
                for (int x = 0; x < 224; x++) {
                    int pixel = scaledBitmap.getPixel(x, y);
                    input[0][y][x][0] = ((pixel >> 16) & 0xFF) / 255.0f;  // Red
                    input[0][y][x][1] = ((pixel >> 8) & 0xFF) / 255.0f;   // Green
                    input[0][y][x][2] = (pixel & 0xFF) / 255.0f;          // Blue
                }
            }

            // Prepare the output array
            float[][] output = new float[1][5];  // 5 classes

            // Run inference
            tfliteInterpreter.run(input, output);

            // Postprocess the output to get the predicted class and confidence
            int predictedClass = -1;
            float maxConfidence = -1;
            for (int i = 0; i < output[0].length; i++) {
                if (output[0][i] > maxConfidence) {
                    maxConfidence = output[0][i];
                    predictedClass = i;
                }
            }

            // Define class labels
            String[] classLabels = {"D1-MC31", "D2-WS50", "D3-ET45", "D4-WT64", "D5-ZEC500"};
            String result = "Prediction: " + classLabels[predictedClass] + "\nConfidence: " + (maxConfidence * 100) + "%";

            // Display the result
            predictionTextView.setText(result);

        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Error running inference!", Toast.LENGTH_SHORT).show();
        }
    }
}