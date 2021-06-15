// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  ModelHandler.swift
//  ORTObjectDetection
//

import AVFoundation
import Accelerate
import CoreImage
import Darwin
import Foundation
import UIKit


/**Result struct*/
struct Result {
    let processTimeMs: Double
    let inferences: [Inference]
}

struct Inference {
    let score: Float
    let className: String
    let rect: CGRect
    let displayColor: UIColor
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

enum OrtModelError : Error {
    case error(_ message: String)
}

class ModelHandler: NSObject {
    
    // MARK: - Inference Properties
    let threadCount: Int32
    let threshold: Float = 0.5
    let threadCountLimit = 10
    
    // MARK: - Model Parameters
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 300
    let inputHeight = 300
    
    private let colors = [
        UIColor.red,
        UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
        UIColor.green,
        UIColor.orange,
        UIColor.blue,
        UIColor.purple,
        UIColor.magenta,
        UIColor.yellow,
        UIColor.cyan,
        UIColor.brown
    ]
    
    private var labels: [String] = []
    
    init?(threadCount: Int32 = 1) {
        self.threadCount = threadCount
        
        super.init()
    }
    /**
     This methods preprocess the image,  runs the ort inferencesession and processes the result
     */
    func runModel(onFrame pixelBuffer: CVPixelBuffer, modelFileInfo: FileInfo, labelsFileInfo: FileInfo) throws -> Result? {
        
        let modelFilename = modelFileInfo.name
        
        labels = loadLabels(fileInfo: labelsFileInfo)
        
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to get model file path with name: \(modelFilename).")
            return nil
        }
        
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
                sourcePixelFormat == kCVPixelFormatType_32BGRA ||
                sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        let imageChannels = 4
        assert(imageChannels >= inputChannels)
        
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        ///preprocess the image
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let scaledPixelBuffer = preprocess(ofSize: scaledSize, pixelBuffer) else {
            return nil
        }
        
        ///Start the ORT inference environment
        let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
        let options = try ORTSessionOptions()
        try options.setLogSeverityLevel(ORTLoggingLevel.verbose)
        try options.setIntraOpNumThreads(self.threadCount) // TODO: check if calling the right methods
        
        let session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        
        let interval: TimeInterval
        
        let inputName = "normalized_input_image_tensor"
        
        guard let rgbData = rgbDataFromBuffer(
            scaledPixelBuffer,
            byteCount: batchSize * inputWidth * inputHeight * inputChannels
        ) else {
            print("Failed to convert the image buffer to RGB data.")
            return nil
        }
        
        let inputTensor = try ORTValue(tensorData: NSMutableData(data: rgbData),
                                       elementType: ORTTensorElementDataType.float,
                                       shape: [1, 300, 300, 3])
        /// Run ORT InferenceSession
        let startDate = Date()
        let outputs = try session.run(withInputs: [inputName: inputTensor],
                                      outputNames: ["TFLite_Detection_PostProcess",
                                                    "TFLite_Detection_PostProcess:1",
                                                    "TFLite_Detection_PostProcess:2",
                                                    "TFLite_Detection_PostProcess:3"],
                                      runOptions: try ORTRunOptions())
        interval = Date().timeIntervalSince(startDate) * 1000
        
        guard let rawOutputValue = outputs["TFLite_Detection_PostProcess"] else {
            throw OrtModelError.error("failed to get model output_0")
        }
        
        let rawOutputData = try rawOutputValue.tensorData() as Data
        
        guard let outputArr: [Float32] = arrayCopiedFromData(rawOutputData) else {
            throw OrtModelError.error("failed to copy output data_0")
        }
        
        guard let rawOutputValue_1 = outputs["TFLite_Detection_PostProcess:1"] else {
            throw OrtModelError.error("failed to get model output_1")
        }
        let rawOutputData_1 = try rawOutputValue_1.tensorData() as Data
        guard let outputArr_1: [Float32] = arrayCopiedFromData(rawOutputData_1) else {
            throw OrtModelError.error("failed to copy output data_1")
        }
        
        guard let rawOutputValue_2 = outputs["TFLite_Detection_PostProcess:2"] else {
            throw OrtModelError.error("failed to get model output_2")
        }
        let rawOutputData_2 = try rawOutputValue_2.tensorData() as Data
        guard let outputArr_2: [Float32] = arrayCopiedFromData(rawOutputData_2) else {
            throw OrtModelError.error("failed to copy output data_2")
        }
        
        guard let rawOutputValue_3 = outputs["TFLite_Detection_PostProcess:3"] else {
            throw OrtModelError.error("failed to get model output_3")
        }
        let rawOutputData_3 = try rawOutputValue_3.tensorData() as Data
        guard let outputArr_3: [Float32] = arrayCopiedFromData(rawOutputData_3) else {
            throw OrtModelError.error("failed to copy output data_3")
        }
        
        // Output order of ssd mobileNet model: detection boxes/classes/scores/num_detection
        let detectionBoxes = outputArr
        let detectionClasses = outputArr_1
        let detectionScores = outputArr_2
        let numDetections:Int = Int(outputArr_3[0])
        
        // Format the results
        let resultArray = formatResults(detectionBoxes: detectionBoxes, detectionClasses: detectionClasses, detectionScores: detectionScores, numDetections: numDetections, width: CGFloat(imageWidth), height: CGFloat(imageHeight))
        
        // Return ORT SessionRun result
        return Result(processTimeMs: interval, inferences: resultArray)
    }
    
    // MARK: - Helper Methods
    func formatResults(detectionBoxes: [Float32], detectionClasses: [Float32], detectionScores: [Float32], numDetections: Int, width: CGFloat, height: CGFloat) -> [Inference] {
        
        var resultsArray: [Inference] = []
        
        if (numDetections == 0) {
            return resultsArray
        }
        
        for i in 0...numDetections - 1 {
            
            let score = detectionScores[i]
            
            // Filter results with score < threshold.
            guard score >= threshold else {
                continue
            }
            
            let detectionClassIndex = Int(detectionClasses[i])
            let detectionClass = labels[detectionClassIndex + 1]
            
            var rect: CGRect = CGRect.zero
            
            // Translate the detected bounding box to CGRect.
            rect.origin.y = CGFloat(detectionBoxes[4*i])
            rect.origin.x = CGFloat(detectionBoxes[4*i+1])
            rect.size.height = CGFloat(detectionBoxes[4*i+2]) - rect.origin.y
            rect.size.width = CGFloat(detectionBoxes[4*i+3]) - rect.origin.x
            
            let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))
            
            let colorToAssign = colorForClass(withIndex: detectionClassIndex + 1)
            let inference = Inference(score: score,
                                      className: detectionClass,
                                      rect: newRect,
                                      displayColor: colorToAssign)
            resultsArray.append(inference)
        }
        
        // Sort results in descending order of confidence.
        resultsArray.sort { (first, second) -> Bool in
            return first.score  > second.score
        }
        
        return resultsArray
    }
    
    /// Preprocess the image by cropping pixel buffer to biggest square and scaling the cropped image to model dimensions.
    private func preprocess(
        ofSize size: CGSize,
        _ buffer: CVPixelBuffer
    ) -> CVPixelBuffer? {
        
        let imageWidth = CVPixelBufferGetWidth(buffer)
        let imageHeight = CVPixelBufferGetHeight(buffer)
        let pixelBufferType = CVPixelBufferGetPixelFormatType(buffer)
        
        assert(pixelBufferType == kCVPixelFormatType_32BGRA ||
                pixelBufferType == kCVPixelFormatType_32ARGB)
        
        let inputImageRowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let imageChannels = 4
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        // Find the biggest square in the pixel buffer and advance rows based on it.
        guard let inputBaseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        // Get vImage_buffer
        var inputVImageBuffer = vImage_Buffer(
            data: inputBaseAddress, height: UInt(imageHeight), width: UInt(imageWidth),
            rowBytes: inputImageRowBytes)
        
        let scaledRowBytes = Int(size.width) * imageChannels
        guard  let scaledImageBytes = malloc(Int(size.height) * scaledRowBytes) else {
            return nil
        }
        var scaledVImageBuffer = vImage_Buffer(data: scaledImageBytes, height: UInt(size.height), width: UInt(size.width), rowBytes: scaledRowBytes)
        
        // Perform the scale operation on input image buffer and store it in scaled vImage buffer.
        let scaleError = vImageScale_ARGB8888(&inputVImageBuffer, &scaledVImageBuffer, nil, vImage_Flags(0))
        
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        guard scaleError == kvImageNoError else {
            return nil
        }
        
        let releaseCallBack: CVPixelBufferReleaseBytesCallback = {mutablePointer, pointer in
            
            if let pointer = pointer {
                free(UnsafeMutableRawPointer(mutating: pointer))
            }
        }
        
        var scaledPixelBuffer: CVPixelBuffer?
        
        // Convert the scaled vImage buffer to CVPixelBuffer
        let conversionStatus = CVPixelBufferCreateWithBytes(
            nil, Int(size.width), Int(size.height), pixelBufferType, scaledImageBytes,
            scaledRowBytes, releaseCallBack, nil, nil, &scaledPixelBuffer)
        
        guard conversionStatus == kCVReturnSuccess else {
            free(scaledImageBytes)
            return nil
        }
        
        return scaledPixelBuffer
    }
    
    
    private func loadLabels(fileInfo: FileInfo) -> [String] {
        var labelData: [String] = []
        let filename = fileInfo.name
        let fileExtension = fileInfo.extension
        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
            fatalError("Labels file not found in bundle. Please add a labels file with name " +
                        "\(filename).\(fileExtension)")
        }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            labelData = contents.components(separatedBy: .newlines)
        } catch {
            fatalError("Labels file named \(filename).\(fileExtension) cannot be read.")
        }
        return labelData
    }
    
    
    private func colorForClass(withIndex index: Int) -> UIColor {
        
        // Assign variations to the base colors for each object based on its index.
        let baseColor = colors[index % colors.count]
        
        var colorToAssign = baseColor
        
        let percentage = CGFloat((10 / 2 - index / colors.count) * 10)
        
        if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
            colorToAssign = modifiedColor
        }
        
        return colorToAssign
    }
    
    
    /// Return  the RGB data representation of the given image buffer.
    func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool = false
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)
        
        let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)
        
        switch (pixelBufferFormat) {
        case kCVPixelFormatType_32BGRA:
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32ARGB:
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32RGBA:
            vImageConvert_RGBA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        default:
            // Unknown pixel format.
            return nil
        }
        
        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        // MARK: [TODO] Consider quantized model here
        // Not quantized, convert to floats
        let bytes = Array<UInt8>(unsafeData: byteData)!
        var floats = [Float]()
        for i in 0..<bytes.count {
            floats.append(Float(bytes[i]) / 255.0)
        }
        return Data(copyingBufferOf: floats)
    }
    
}



func arrayCopiedFromData<T>(_ data: Data) -> [T]? {
    guard data.count % MemoryLayout<T>.stride == 0 else { return nil }
    
    return data.withUnsafeBytes {
        bytes -> [T] in
        return Array(bytes.bindMemory(to: T.self))
    }
}

// MARK: - Extensions
extension Data {
    /// Creates a new buffer by copying the buffer pointer of the given array.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    /// Creates a new array from the bytes of the given unsafe data.
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
        #if swift(>=5.0)
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
        #else
        self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
                start: $0,
                count: unsafeData.count / MemoryLayout<Element>.stride
            ))
        }
        #endif  // swift(>=5.0)
    }
}

extension UIColor {
    
    /**This method returns colors modified by percentage value of color represented by the current object.*/
    func getModified(byPercentage percent: CGFloat) -> UIColor? {
        
        var red: CGFloat = 0.0
        var green: CGFloat = 0.0
        var blue: CGFloat = 0.0
        var alpha: CGFloat = 0.0
        
        guard self.getRed(&red, green: &green, blue: &blue, alpha: &alpha) else {
            return nil
        }
        
        // Returns the color comprised by percentage r g b values of the original color.
        let colorToReturn = UIColor(displayP3Red: min(red + percent / 100.0, 1.0), green: min(green + percent / 100.0, 1.0), blue: min(blue + percent / 100.0, 1.0), alpha: 1.0)
        
        return colorToReturn
    }
}
