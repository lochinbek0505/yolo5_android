package com.example.yolotest

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.SurfaceView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import com.example.yolotest.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.DataType
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: SurfaceView
    private lateinit var tflite: Interpreter
    private lateinit var labels: List<String>
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var binding: ActivityMainBinding
    private val executor = Executors.newSingleThreadExecutor()
    private val imageSize = Size(416, 416)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        previewView = binding.previewView
        overlay = binding.overlay
        overlay.setZOrderOnTop(true)
        overlay.holder.setFormat(PixelFormat.TRANSPARENT)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            loadModelAndLabels()
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        }
    }

    private fun loadModelAndLabels() {
        val modelFile = FileUtil.loadMappedFile(this, "model.tflite")
        tflite = Interpreter(modelFile)
        labels = FileUtil.loadLabels(this, "labels.txt")

        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(416, 416, ResizeOp.ResizeMethod.BILINEAR))
            .build()
    }

    private fun runInference(bitmap: Bitmap): List<DetectionResult> {
        val resized = bitmap.scale(416, 416)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resized)

        val input = imageProcessor.process(tensorImage).buffer

        // Modelga mos chiqish shakli — modelga qarab o'zgartiring!
        val output = Array(1) { Array(10647) { FloatArray(7) } }

        tflite.run(input, output)

        val results = mutableListOf<DetectionResult>()
        val confidenceThreshold = 0.5f

        for (i in 0 until 2535) {
            val row = output[0][i]
            val x = row[0]
            val y = row[1]
            val w = row[2]
            val h = row[3]
            val objConf = row[4]
            val classId = row[5].toInt()
            val classConf = row[6]
            val finalConf = objConf * classConf
            val label = labels.getOrElse(classId) { "Unknown" }

            Log.e("TEST_ONAGNI_EMGIR","$objConf * $classConf $finalConf $label")
            if (finalConf > confidenceThreshold) {
                val label = labels.getOrElse(classId) { "Unknown" }
                results.add(DetectionResult(x, y, w, h, finalConf, label))
            }
        }

        return results
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(imageSize)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { imageProxy ->
                val bitmap = imageProxyToBitmap(imageProxy)
                val results = runInference(bitmap)
                runOnUiThread {
                    drawResults(results)
                }
                imageProxy.close()
            })

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun drawResults(results: List<DetectionResult>) {
        val canvas = overlay.holder.lockCanvas() ?: return
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 4f
            textSize = 40f
            isAntiAlias = true
        }

        if (results.isEmpty()) {
            overlay.holder.unlockCanvasAndPost(canvas)
            return
        }

        val best = results.maxByOrNull { it.confidence } ?: return

        val modelInputSize = 416f
        val scaleX = overlay.width / modelInputSize
        val scaleY = overlay.height / modelInputSize

        val left = (best.x - best.w / 2) * scaleX
        val top = (best.y - best.h / 2) * scaleY
        val right = (best.x + best.w / 2) * scaleX
        val bottom = (best.y + best.h / 2) * scaleY

        canvas.drawRect(left, top, right, bottom, paint)
        canvas.drawText("${best.label} ${"%.2f".format(best.confidence)}", left, top - 10, paint)

        overlay.holder.unlockCanvasAndPost(canvas)
    }

    data class DetectionResult(
        val x: Float,
        val y: Float,
        val w: Float,
        val h: Float,
        val confidence: Float,
        val label: String
    )

    override fun onDestroy() {
        super.onDestroy()
        tflite.close()
    }
}
