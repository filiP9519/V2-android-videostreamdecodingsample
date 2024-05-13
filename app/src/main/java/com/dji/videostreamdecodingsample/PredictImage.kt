package com.dji.videostreamdecodingsample


import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.dji.videostreamdecodingsample.ml.Model
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PredictImage : AppCompatActivity() {

    private lateinit var cameraButton: Button
    private lateinit var galleryButton: Button
    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private var imageSize = 100 // Assuming imageSize is defined

    // Tracking class counts
    private val classCounts = IntArray(5)
    private var totalClassifications = 0
    private val classes = arrayOf("Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy", "Apple_Powdery_mildew", "Apple_scab")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_predict_image)

        //cameraButton = findViewById(R.id.button)
        galleryButton = findViewById(R.id.button2)
        resultTextView = findViewById(R.id.result)
        imageView = findViewById(R.id.imageView)

        val backToMainActivity = findViewById<Button>(R.id.activity_back_to_main)
        backToMainActivity.setOnClickListener {
            val Intent = Intent(this, MainActivity::class.java)
            startActivity(Intent)

        }
/*
        cameraButton.setOnClickListener {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                startActivityForResult(cameraIntent, 3)
            } else {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
            }
        }
*/
        galleryButton.setOnClickListener {
            val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI).apply {
                putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            }
            startActivityForResult(galleryIntent, 1)
        }
    }


    private fun classifyImage(image: Bitmap) {
        try {
            val model = Model.newInstance(applicationContext)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)
            val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3).apply {
                order(ByteOrder.nativeOrder())
            }

            val intValues = IntArray(imageSize * imageSize)
            image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)
            intValues.forEach { pixel ->
                byteBuffer.putFloat(((pixel shr 16) and 0xFF) * (1f / 1))
                byteBuffer.putFloat(((pixel shr 8) and 0xFF) * (1f / 1))
                byteBuffer.putFloat((pixel and 0xFF) * (1f / 1))
            }

            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.getOutputFeature0AsTensorBuffer()

            val confidences = outputFeature0.floatArray
            val maxPos = confidences.indices.maxByOrNull { confidences[it] } ?: 0
            val maxConfidence = confidences[maxPos]

            // Update class counts
            classCounts[maxPos]++
            totalClassifications++

            model.close()
        } catch (e: IOException) {
            // Handle the exception
            Log.e("MainActivity", "Classify image error: ${e.message}")
        }
    }

    private fun displayResults() {
        val resultText = StringBuilder()
        for (i in classes.indices) {
            val percentage = (classCounts[i].toDouble() / totalClassifications) * 100
            resultText.append("${classes[i]}: ${"%.2f".format(percentage)}%\n")
        }
        resultTextView.text = resultText.toString()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {
                val image = data?.extras?.get("data") as? Bitmap ?: return
                val dimension = minOf(image.width, image.height)
                val thumbnail = Bitmap.createScaledBitmap(image, dimension, dimension, false)
                imageView.setImageBitmap(thumbnail)

                val resizedImage = Bitmap.createScaledBitmap(thumbnail, imageSize, imageSize, false)
                classifyImage(resizedImage)
            } else if (requestCode == 1) {
                data?.clipData?.let { clipData ->
                    // Multiple images selected
                    for (i in 0 until clipData.itemCount) {
                        val uri = clipData.getItemAt(i).uri
                        val image = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                        val resizedImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
                        classifyImage(resizedImage)
                    }
                } ?: run {
                    // Single image selected
                    val uri = data?.data ?: return
                    val image = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                    imageView.setImageBitmap(image)

                    val resizedImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, false)
                    classifyImage(resizedImage)
                }

                // Display results after processing all images
                displayResults()
            }
        }
    }

}
