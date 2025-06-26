package com.example.yolotest

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.SurfaceView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.yolotest.ml.Model
import com.example.yolotest.ml.YangiModel
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: SurfaceView
    private lateinit var labels: List<String>
    private lateinit var model: YangiModel
    private lateinit var imageProcessor: ImageProcessor

    private val executor = Executors.newSingleThreadExecutor()
    private val imageSize = Size(416, 416)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlay)
        overlay.setZOrderOnTop(true)
        overlay.holder.setFormat(PixelFormat.TRANSPARENT)

        if (allPermissionsGranted()) {
            loadModelAndLabels()
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        }
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        baseContext, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            loadModelAndLabels()
            startCamera()
        } else {
            Toast.makeText(this, "Kamera ruxsati kerak", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadModelAndLabels() {
        model = YangiModel.newInstance(this)
        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(416, 416, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()
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
//                drawResults(results)
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

    private fun runInference(bitmap: Bitmap): List<DetectionResult> {
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val outputs = model.process(processedImage.tensorBuffer)
        val outputArray = outputs.outputFeature0AsTensorBuffer.floatArray

        val results = mutableListOf<DetectionResult>()
        val numDetections = 10647

        for (i in 0 until numDetections) {
            val offset = i * 7
            val conf = outputArray[offset + 4]
            if (conf > 0.5f) {
                val x = outputArray[offset]
                val y = outputArray[offset + 1]
                val w = outputArray[offset + 2]
                val h = outputArray[offset + 3]
                val classScores = outputArray.copyOfRange(offset + 5, offset + 7)
                val classId = classScores.withIndex().maxByOrNull { it.value }?.index ?: -1
                Log.d("MainActivitydfdx", "Class ID: ${labels.getOrElse(classId) { "?" }}")
                results.add(DetectionResult(x, y, w, h, conf, labels.getOrElse(classId) { "?" }))
            }
        }

        return results
    }

    private fun drawResults(results: List<DetectionResult>) {
        val canvas = overlay.holder.lockCanvas() ?: return
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 4f
            textSize = 40f
        }

        val scaleX = overlay.width.toFloat()
        val scaleY = overlay.height.toFloat()

        results.forEach {
            val left = (it.x - it.w / 2) * scaleX
            val top = (it.y - it.h / 2) * scaleY
            val right = (it.x + it.w / 2) * scaleX
            val bottom = (it.y + it.h / 2) * scaleY
            canvas.drawRect(left, top, right, bottom, paint)
            canvas.drawText("${it.label} ${"%.2f".format(it.confidence)}", left, top - 10, paint)
        }

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
        model.close()
    }
}
