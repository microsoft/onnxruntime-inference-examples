#!/bin/bash

export ANDROID_HOME=`realpath ~/android_sdk/`
export LOCAL_ORT_AAR_PATH='/home/guangyunhan/workspaces/onnxruntime/build/android_aar/aar_out/Release/com/microsoft/onnxruntime/onnxruntime-mobile/1.11.0/onnxruntime-mobile-1.11.0.aar'

set -ex

./gradlew $@
