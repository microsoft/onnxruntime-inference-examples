/**
 * Copyright 2019 The TensorFlow Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Portions Copyright (c) Microsoft Corporation

import UIKit

class ViewController: UIViewController {
    @IBOutlet var previewView: PreviewView!
    @IBOutlet var overlayView: OverlayView!
    @IBOutlet var bottomSheetView: UIView!
    
    private let edgeOffset: CGFloat = 2.0
    private let displayFont = UIFont.systemFont(ofSize: 14.0, weight: .medium)
    
    private var result: Result?
    private var previousInferenceTimeMs: TimeInterval = Date.distantPast.timeIntervalSince1970 * 1000
    private let delayBetweenInferencesMs: Double = 1000
    
    // Handle all the camera related functionality
    private lazy var cameraCapture = CameraManager(previewView: previewView)
    
    // Handle the presenting of results on the screen
    private var inferenceViewController: InferenceViewController?
    
    // Handle all model and data preprocessing and run inference
    private var modelHandler: ModelHandler? = ModelHandler(
        modelFileInfo: (name: "ssd_mobilenet_v1", extension: "ort"),
        labelsFileInfo: (name: "labelmap", extension: "txt"))
    
    // MARK: View Controller Life Cycle

    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard modelHandler != nil else {
            fatalError("Model set up failed")
        }
        
        cameraCapture.delegate = self
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        cameraCapture.checkCameraConfigurationAndStartSession()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cameraCapture.stopSession()
    }
    
    // MARK: Storyboard Segue Handlers

    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        super.prepare(for: segue, sender: sender)
        
        if segue.identifier == "EMBED" {
            guard let tempModelHandler = modelHandler else {
                return
            }
            inferenceViewController = segue.destination as? InferenceViewController
            inferenceViewController?.wantedInputHeight = tempModelHandler.inputHeight
            inferenceViewController?.wantedInputWidth = tempModelHandler.inputWidth
            inferenceViewController?.threadCountLimit = tempModelHandler.threadCountLimit
            inferenceViewController?.currentThreadCount = tempModelHandler.threadCount
            inferenceViewController?.delegate = self
            
            guard let tempResult = result else {
                return
            }
            inferenceViewController?.inferenceTime = tempResult.processTimeMs
        }
    }
}

// MARK: InferenceViewControllerDelegate Methods

extension ViewController: InferenceViewControllerDelegate {
    func didChangeThreadCount(to count: Int32) {
        if modelHandler?.threadCount == count { return }
        modelHandler = ModelHandler(modelFileInfo: (name: "ssd_mobilenet_v1", extension: "ort"),
                                    labelsFileInfo: (name: "labelmap", extension: "txt"),
                                    threadCount: count)
    }
}

// MARK: CameraManagerDelegate Methods

extension ViewController: CameraManagerDelegate {
    func didOutput(pixelBuffer: CVPixelBuffer) {
        runModel(onPixelBuffer: pixelBuffer)
    }
    
    // MARK: Session Handling Alerts

    func presentCameraPermissionsDeniedAlert() {
        let alertController = UIAlertController(title: "Camera Permissions Denied",
                                                message: "Camera permissions have been denied for this app.",
                                                preferredStyle: .alert)
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        let settingsAction = UIAlertAction(title: "Settings", style: .default) { _ in
            UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!,
                                      options: [:],
                                      completionHandler: nil)
        }
        alertController.addAction(cancelAction)
        alertController.addAction(settingsAction)
        
        present(alertController, animated: true, completion: nil)
    }
    
    func presentVideoConfigurationErrorAlert() {
        let alert = UIAlertController(title: "Camera Configuration Failed",
                                      message: "There was an error while configuring camera.",
                                      preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        
        present(alert, animated: true)
    }
    
    func runModel(onPixelBuffer pixelBuffer: CVPixelBuffer) {
        let currentTimeMs = Date().timeIntervalSince1970 * 1000
        guard (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs
        else { return }
        previousInferenceTimeMs = currentTimeMs
        
        result = try! self.modelHandler?.runModel(onFrame: pixelBuffer)
        
        guard let displayResult = result else {
            return
        }
        
        // Display results by the `InferenceViewController`.
        DispatchQueue.main.async {
            let resolution = CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                                    height: CVPixelBufferGetHeight(pixelBuffer))
            self.inferenceViewController?.resolution = resolution
            
            var inferenceTime: Double = 0
            if let resultInferenceTime = self.result?.processTimeMs {
                inferenceTime = resultInferenceTime
            }
            self.inferenceViewController?.inferenceTime = inferenceTime
            self.inferenceViewController?.tableView.reloadData()
            
            // Draw bounding boxes and compute the inference score
            self.drawBoundingBoxesAndCalculate(onInferences: displayResult.inferences,
                                               withImageSize: CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                                                                     height: CVPixelBufferGetHeight(pixelBuffer)))
        }
    }
    
    func drawBoundingBoxesAndCalculate(onInferences inferences: [Inference], withImageSize imageSize: CGSize) {
        overlayView.objectOverlays = []
        overlayView.setNeedsDisplay()
        
        guard !inferences.isEmpty else {
            return
        }
        
        var objectOverlays: [ObjectOverlay] = []
        
        for inference in inferences {
            // Translate the bounding box rectangle to the current view
            var convertedRect = inference.rect.applying(
                CGAffineTransform(
                    scaleX: overlayView.bounds.size.width / imageSize.width,
                    y: overlayView.bounds.size.height / imageSize.height))
            
            if convertedRect.origin.x < 0 {
                convertedRect.origin.x = edgeOffset
            }
            
            if convertedRect.origin.y < 0 {
                convertedRect.origin.y = edgeOffset
            }
            
            if convertedRect.maxY > overlayView.bounds.maxY {
                convertedRect.size.height = overlayView.bounds.maxY - convertedRect.origin.y - edgeOffset
            }
            
            if convertedRect.maxX > overlayView.bounds.maxX {
                convertedRect.size.width = overlayView.bounds.maxX - convertedRect.origin.x - edgeOffset
            }
            
            let scoreValue = Int(inference.score * 100.0)
            let string = "\(inference.className)  (\(scoreValue)%)"
            
            let nameStringsize = string.size(usingFont: displayFont)
            
            let objectOverlay = ObjectOverlay(name: string,
                                              borderRect: convertedRect,
                                              nameStringSize: nameStringsize,
                                              color: inference.displayColor,
                                              font: displayFont)
            
            objectOverlays.append(objectOverlay)
        }
        
        // Update overlay view with detected bounding boxes and class names.
        draw(objectOverlays: objectOverlays)
    }
    
    func draw(objectOverlays: [ObjectOverlay]) {
        overlayView.objectOverlays = objectOverlays
        overlayView.setNeedsDisplay()
    }
}

extension String {
    /// This method gets size of a string with a particular font.
    func size(usingFont font: UIFont) -> CGSize {
        let attributedString = NSAttributedString(string: self, attributes: [NSAttributedString.Key.font: font])
        return attributedString.size()
    }
}
