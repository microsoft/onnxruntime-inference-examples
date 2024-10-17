plugins {
    id("com.android.application")
}

android {
    namespace = "ai.onnxruntime.genai.demo"
    compileSdk = 33

    defaultConfig {
        applicationId = "ai.onnxruntime.genai.demo"
        minSdk = 27
        targetSdk = 33
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            //noinspection ChromeOsAbiSupport
            //abiFilters += listOf("arm64-v8a", "x86_64")
            abiFilters += listOf("arm64-v8a")
        }
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
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    buildFeatures {
        viewBinding = true
    }

    // set this so QNN libs will show up in nativeLibraryDir
    packaging.jniLibs.useLegacyPackaging = true
}

dependencies {

    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.9.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")

    // ONNX Runtime with GenAI
    //implementation("com.microsoft.onnxruntime:onnxruntime-android:latest.release")
    implementation(files("libs/onnxruntime-android-qnn-1.20.0.aar"))
    implementation(files("libs/onnxruntime-genai-android-0.5.0-dev.aar"))
    implementation("com.qualcomm.qti:qnn-runtime:2.27.0")

}