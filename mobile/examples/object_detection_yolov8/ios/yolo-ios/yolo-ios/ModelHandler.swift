//
//  ModelHandler.swift
//  PHPickerDemo
//
//  Created by Jayashree Patil on 27/11/23.
//  Copyright © 2023 Apple. All rights reserved.
//

import Foundation
import onnxruntime_objc
import UIKit
import CoreImage
import Accelerate.vImage
import CoreML

//  struct to shape data
struct ShapeData<T> {
    var shape: [Int]
    var data: [T]
    
    subscript(indices: Int...) -> T? {
        guard indices.count == shape.count else {
            print("Invalid number of indices provided")
            return nil
        }
        var index = 0
        var stride = 1
        for (dimensionIndex, dimensionSize) in zip(indices, shape).reversed() {
            if dimensionIndex >= dimensionSize || dimensionIndex < 0 {
                print("Index out of range")
                return nil
            }
            index += dimensionIndex * stride
            stride *= dimensionSize
        }
        return data[index]
    }
    
    subscript(index: Int) -> [T]? {
        guard index < shape[0] else {
            print("Index out of range for accessing row")
            return nil
        }
        let rowStart = index * shape[1] * shape[2]
        let rowEnd = (index + 1) * shape[1] * shape[2]
        return Array(data[rowStart..<rowEnd])
    }
    
    subscript(row: Int, column: Int) -> T? {
        let index = row * shape[2] + column
        guard index < shape[1] * shape[2] else {
            print("Index out of range for accessing column")
            return nil
        }
        return data[row * shape[1] * shape[2] + column]
    }
    
    // get total row count
    func getRowCount() -> Int? {
        return shape.count > 1 ? shape[1] : nil
    }
    
    // get total columnn count
    func getColCount() -> Int? {
        return shape.count > 1 ? shape[2] : nil
    }
}

// detection object struct
struct DetectionObject {
    var className:  String
    var score: Float?
    var bounds: [Float?]
}

enum OrtModelError: Error {
    case error(_ message: String)
}

class ModelHandler: NSObject {
    
    // MARK: - Inference Properties
    let threadCount: Int32
    let threshold: Float = 0.5
    let iouThreshold = 0.45
    
    // MARK: - Model Parameters
    let inputWidth = 640
    let inputHeight = 640

    private var labels: [String] = []
    
    // Information about a model file or labels file.
    typealias FileInfo = (name: String, extension: String)
    
    // ORT inference session and environment object for performing inference on the given ssd model
    private var session: ORTSession
    private var env: ORTEnv
    
    // MARK: - Initialization of ModelHandler
    init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int32 = 1) {
       
        // store model file name
        let modelFilename = modelFileInfo.name
        
        // store model file path
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to get model file path with name: \(modelFilename).")
            return nil
        }
      
        self.threadCount = threadCount
        
        // creating ORT session
        do {
            // Start the ORT inference environment and specify the options for session
            env = try ORTEnv(loggingLevel: ORTLoggingLevel.verbose)
            let options = try ORTSessionOptions()
            try options.setLogSeverityLevel(ORTLoggingLevel.verbose)
            try options.setIntraOpNumThreads(threadCount)
            
            // Create the ORTSession
            session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        } catch let error as NSError{
            print("Failed to create ORTSession")
            print(error)
            return nil
        }
        
        super.init()
        
        labels = loadLabels(fileInfo: labelsFileInfo)
    }
    
    // This method preprocesses the image, runs the ort inferencesession and returns the tensor result
    func runModel(cgImage: CGImage) {
        // scale image to 640 * 640
        guard let image =  resizeCgImage(cgImage) else { return }

        guard let format = vImage_CGImageFormat(bitsPerComponent: 8, bitsPerPixel: 24, colorSpace: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)) else { return }
        
        var buffer = try! vImage_Buffer(cgImage: image, format: format)
        
        defer {
            buffer.free()
        }
        
        let height = Int(buffer.height)
        
        let width = Int(buffer.width)
        
        let bufData = buffer.data.assumingMemoryBound(to: UInt8.self)
        for i in 0..<height {
            
            let rowOffset = i * buffer.rowBytes
            
            for j in 0..<width {
                let pixelOffset = rowOffset + j * 3
                
                // Accessing individual bytes for RGBA channels
                let red = bufData[pixelOffset]
                let green = bufData[pixelOffset + 1]
                let blue = bufData[pixelOffset + 2]
                
            }
        }
        
        let alignmentAndRowBytes = try! vImage_Buffer.preferredAlignmentAndRowBytes(
            width: width,
            height: height,
            bitsPerPixel: 8)

        let planarData = UnsafeMutableRawPointer.allocate(
            byteCount: alignmentAndRowBytes.rowBytes * height * 3,
            alignment: alignmentAndRowBytes.alignment)
            defer {
        //      planarData.deallocate()
            }
        
        var Rbuffer = vImage_Buffer(data: planarData,
                                    height: buffer.height,
                                    width: buffer.width,
                                    rowBytes: alignmentAndRowBytes.rowBytes)
        
        var Gbuffer = vImage_Buffer(data: planarData.advanced(by: alignmentAndRowBytes.rowBytes * height * 1),
                                    height: buffer.height,
                                    width: buffer.width,
                                    rowBytes: alignmentAndRowBytes.rowBytes)
        
        var Bbuffer = vImage_Buffer(data: planarData.advanced(by: alignmentAndRowBytes.rowBytes * height * 2),
                                    height: buffer.height,
                                    width: buffer.width,
                                    rowBytes: alignmentAndRowBytes.rowBytes)
        
        let error = vImageConvert_RGB888toPlanar8(&buffer, &Rbuffer, &Gbuffer, &Bbuffer, vImage_Flags(kvImageNoFlags))
        guard error == kvImageNoError else {
            fatalError("Error in vImageConvert_RGB888toPlanar8: \(error)")
        }
        
        let floatData = UnsafeMutablePointer<Float>.allocate(capacity: alignmentAndRowBytes.rowBytes * height * 3)
        
        for i in 0..<alignmentAndRowBytes.rowBytes * height * 3 {
            floatData[i] = Float(planarData.load(fromByteOffset: i, as: UInt8.self))/Float(255)
        }
        
        
        let data = NSMutableData(bytes: floatData, length: alignmentAndRowBytes.rowBytes * height * 3 * 4)
        
        let tensor = try! ORTValue(tensorData: data, elementType: ORTTensorElementDataType.float, shape: [1, 3, 640, 640])
            
        // name of the tensor model input, which we will use later while creting a tensor
        let inputName = "images"
        
        do{
            // get output result
            let outputs = try self.session.run(withInputs: [inputName: tensor],
                                               outputNames: ["output0"],
                                               runOptions: nil)
            
            // store output data
            guard let rawOutputValue = outputs["output0"] else {
                throw OrtModelError.error("failed to get model output_0")
            }
            
            // get shape of the tensor data
            let shapeOfOutputValue = try rawOutputValue.tensorTypeAndShapeInfo().shape
 
            // store output data as a raw data
            let rawOutputData = try rawOutputValue.tensorData() as Data

            // convert bytedata to float array
            let floatValues = rawOutputData.withUnsafeBytes { (buffer: UnsafeRawBufferPointer) -> [Float] in
                let floatBuffer = buffer.bindMemory(to: Float.self)
                let count = rawOutputData.count / MemoryLayout<Float>.stride
                return Array(UnsafeBufferPointer(start: floatBuffer.baseAddress, count: count))
            }
            
            
            // shape tensor data and store result
           let resultArray =  shapeTensorData(shape: [1, 84, 8400], data: floatValues, labels : labels, image: image)
            
            // run nonMaxSuppression algorithm
            let finalBoxes = nonMaxSuppression(boxes: resultArray, threshold: Float(iouThreshold))
        
            print("*******************************************************************")
            print("Result : ", finalBoxes)
            
        }catch {
            // Handle the error here
            print("Error occurred: \(error)")
        }
    }
    
    
    // MARK: - Helper Methods
    
    // load labels
    private func loadLabels(fileInfo: FileInfo) -> [String] {
        var labelData: [String] = []
        let filename = fileInfo.name
        let fileExtension = fileInfo.extension
        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
            print("Labels file not found in bundle. Please add a labels file with name " +
                  "\(filename).\(fileExtension)")
            return labelData
        }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            labelData = contents.components(separatedBy: .newlines)
        } catch {
            print("Labels file named \(filename).\(fileExtension) cannot be read.")
        }
        
        return labelData
    }
    
    // function used to resize image to 640 * 640
    func resizeCgImage(_ image: CGImage) -> CGImage? {
        
        guard let colorSpace = image.colorSpace else { return nil }
        
        guard let context = CGContext(data: nil, width: Int(inputWidth), height: Int(inputHeight), bitsPerComponent: image.bitsPerComponent, bytesPerRow: image.bytesPerRow, space: colorSpace, bitmapInfo: image.alphaInfo.rawValue) else { return nil }
               
        // draw image to context (resizing it)
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: Int(inputWidth), height: Int(inputHeight)))
               
        // extract resulting image from context
        return context.makeImage()
        }
}
    
// postproccess - this method shape the tensor result and return bounding boxes with score and class
func shapeTensorData(shape: [Int],data: [Float32], labels: [String], image: CGImage) -> [DetectionObject]{
     
     // create data of the given shape
     let data = ShapeData(shape: shape, data: data)
    
    // Create an array of DetectionObject
    var resultsArray: [DetectionObject] = []

    if let columns = data.getColCount() {
        for colIndex in 0..<columns{
            
           let xAxis = data[0,0,colIndex]
           let  yAxis = data[0,1,colIndex]
           let  width = data[0,2,colIndex]
           let height = data[0,3,colIndex]
            
           // store max score
           var maxScore = Float(0)
            
            // store index of max score
           var maxIndex = 0
            
            if let rows  =  data.getRowCount() {
                for rowIndex in 4..<rows-3{
                    if  let  score = data[0,rowIndex,colIndex] {
                        if(score > maxScore){
                            maxScore = score
                            maxIndex = rowIndex
                        }
                    }else{
                        print("Unable to determine score")
                    }
                    
                }
                
                // get scores greter than 0.25
                if(maxScore > 0.25){
                    // creating bounding box
                    let boundingBox = [xAxis, yAxis, width, height]

                    // store result
                    let result = DetectionObject(className: labels[maxIndex-4], score: maxScore, bounds: boundingBox)
                    
                    resultsArray.append(result)
                }
                
            }else{
                print("Unable to determine the number of rows.")
            }
            
        }
    } else {
        print("Unable to determine the number of columns.")
    }
    
    return resultsArray
}

// calculate IOU
func calculateIOU(_ a: [Float?], _ b: [Float?]) -> Float {
    let areaA = (a[2] ?? 0) * (a[3] ?? 0)
    let areaABottomRightX = (a[0] ?? 0) + (a[2] ?? 0)
    let areaABottomRightY = (a[1] ?? 0) + (a[3] ?? 0)

    if areaA <= 0.0 {
        return 0.0
    }

    let areaB = (b[2] ?? 0) * (b[3] ?? 0)

    if areaB <= 0.0 {
        return 0.0
    }

    let areaBBottomRightX = (b[0] ?? 0) + (b[2] ?? 0)
    let areaBBottomRightY = (b[1] ?? 0) + (b[3] ?? 0)

    let intersectionLeftX = max(a[0] ?? 0, b[0] ?? 0)
    let intersectionLeftY = max(a[1] ?? 0, b[1] ?? 0)
    let intersectionBottomX = min(areaABottomRightX, areaBBottomRightX)
    let intersectionBottomY = min(areaABottomRightY, areaBBottomRightY)

    let intersectionWidth = max(intersectionBottomX - intersectionLeftX, 0)
    let intersectionHeight = max(intersectionBottomY - intersectionLeftY, 0)
    let intersectionArea = intersectionWidth * intersectionHeight

    return intersectionArea / (areaA + areaB - intersectionArea)
}

// used to remove overlapping bounding boxes.
func nonMaxSuppression(boxes: [DetectionObject], threshold: Float) -> [DetectionObject] {
    let newBoxes = boxes.sorted { $0.score ?? 0.0 > $1.score ?? 0.0 }
    
    var selected: [DetectionObject] = []
    var active = [Bool](repeating: true, count: boxes.count)
    var numActive = active.count

    var done = false
    for i in 0..<boxes.count {
        if active[i] {
            let boxA = newBoxes[i]
            selected.append(boxA)

            for j in (i + 1)..<newBoxes.count {
                if active[j] {
                    let boxB = newBoxes[j]

                    if boxA.className == boxB.className {
                        // IOU Implementation
                        let iou = calculateIOU(boxA.bounds, boxB.bounds)
                        
                        if iou > threshold {
                            active[j] = false
                            numActive -= 1
                            
                            if numActive <= 0 {
                                done = true
                                break
                            }
                        }}
                }
            }
            if done {
                break
            }
        }
    }
    return selected
}

// MARK: - Extensions
extension Data {
        // Create a new buffer by copying the buffer pointer of the given array.
        init<T>(copyingBufferOf array: [T]) {
            self = array.withUnsafeBufferPointer(Data.init)
        }
    }
    
extension Array {
        // Create a new array from the bytes of the given unsafe data.
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
#endif // swift(>=5.0)
        }
    }
