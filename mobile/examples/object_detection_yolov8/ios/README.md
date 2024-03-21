## ONNX Runtime Mobile object detection iOS sample application

This sample application makes use of Yolov8 object detection model to identify objects in images and provides identified class, bounding boxes and score as a result.


### Model
We use pre-trained yolov8 model in this sample app. The original yolov8n.pt model can be downloaded. [Here](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt) <br/>
- We converted this yolov8n.pt model to yolov8n.onnx using https://docs.ultralytics.com/modes/export/#usage-examples <br/>
- Generated yolov8n.ort from yolov8n.onnx using https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html#convert-onnx-models-to-ort-format

### Requirements
- Install Xcode 12.5 and above (preferably latest version)

### Build And Run
1. In terminal, run pod install under onnxruntime-object-detection-yolo-ios/yolo-ios to generate the workspace file.
   - At the end of this step, you should get a file called yolo-ios.xcworkspace.
2. Open onnxruntime-object-detection-yolo-ios/yolo-ios/yolo-ios.xcworkspace in Xcode and make sure to select your corresponding development team under Target-General-Signing for a proper codesign 
   procedure to run the app.
3. Connect your iOS device, build and run the app.
4. Select any image for detecting objects.
5. On Xcode console you can see bounding box, class and score as a result.

