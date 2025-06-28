import com.android.tools.r8.internal.iM

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.yolotest"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.yolotest"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    aaptOptions {
        noCompress ("tflite")
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }

    buildFeatures {
        viewBinding=true
        mlModelBinding = true
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)


    implementation("org.tensorflow:tensorflow-lite:2.9.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.3")
    implementation("org.tensorflow:tensorflow-lite-task-vision:0.4.3")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.9.0")
    implementation("org.tensorflow:tensorflow-lite-metadata:0.1.0")

    implementation("androidx.camera:camera-camera2:1.1.0")
    implementation("androidx.camera:camera-lifecycle:1.1.0")
    implementation("androidx.camera:camera-view:1.1.0")

}