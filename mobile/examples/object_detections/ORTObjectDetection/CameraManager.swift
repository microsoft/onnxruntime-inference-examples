// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  CameraManager.swift
//  ORTObjectDetection
//

import UIKit
import AVFoundation

// MARK: CameraFeedManagerDelegate Declaration
protocol CameraManagerDelegate: AnyObject {
    
    /// This method delivers the pixel buffer of the current frame seen by the device's camera.
    func didOutput(pixelBuffer: CVPixelBuffer)
    
    /// This method initimates that the camera permissions have been denied.
    func presentCameraPermissionsDeniedAlert()
    
    /// This method initimates that there was an error in video configurtion.
    func presentVideoConfigurationErrorAlert()
    
}


/**
 This enum holds the state of the camera initialization.
 */
enum CameraConfiguration {
    
    case success
    case failed
    case permissionDenied
}

/**
 This class manages all camera related functionality
 */
class CameraManager: NSObject {
    
    // MARK: Camera Related Instance Variables
    private let session: AVCaptureSession = AVCaptureSession()
    private let previewView: PreviewView
    private let sessionQueue = DispatchQueue(label: "sessionQueue")
    private var cameraConfiguration: CameraConfiguration = .failed
    private lazy var videoDataOutput = AVCaptureVideoDataOutput()
    private var isSessionRunning = false
    
    weak var delegate: CameraManagerDelegate?
    
    init(previewView: PreviewView) {
        self.previewView = previewView
        super.init()
        
        // Initializes the session
        session.sessionPreset = .high
        self.previewView.session = session
        self.previewView.previewLayer.connection?.videoOrientation = .portrait
        self.previewView.previewLayer.videoGravity = .resizeAspectFill
        self.attemptToConfigureSession()
    }
    
    // MARK: Camera Session start and end methods
    func checkCameraConfigurationAndStartSession() {
        sessionQueue.async {
            switch self.cameraConfiguration {
            case .success:
                self.startSession()
            case .failed:
                DispatchQueue.main.async {
                    self.delegate?.presentVideoConfigurationErrorAlert()
                }
            case .permissionDenied:
                DispatchQueue.main.async {
                    self.delegate?.presentCameraPermissionsDeniedAlert()
                }
            }
        }
    }
    
    
    func stopSession() {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
                self.isSessionRunning = self.session.isRunning
            }
        }
    }
    
    private func startSession() {
        self.session.startRunning()
        self.isSessionRunning = self.session.isRunning
    }
    
    // MARK: Camera permission and configuration handling methods
    /**
     This method requests for camera permissions and handles the configuration of the session and stores the result of configuration.
     */
    private func attemptToConfigureSession() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            self.cameraConfiguration = .success
        case .notDetermined:
            self.sessionQueue.suspend()
            self.requestCameraAccess(completion: { (granted) in
                self.sessionQueue.resume()
            })
        case .denied:
            self.cameraConfiguration = .permissionDenied
        default:
            break
        }
        
        self.sessionQueue.async {
            self.configureSession()
        }
    }
    
    
    private func requestCameraAccess(completion: @escaping (Bool) -> ()) {
        AVCaptureDevice.requestAccess(for: .video) { (granted) in
            if !granted {
                self.cameraConfiguration = .permissionDenied
            }
            else {
                self.cameraConfiguration = .success
            }
            completion(granted)
        }
    }
    
    /**
     This method handles all the steps to configure an AVCaptureSession.
     */
    private func configureSession() {
        
        guard cameraConfiguration == .success else {
            return
        }
        session.beginConfiguration()
        
        guard addVideoDeviceInput() == true else {
            self.session.commitConfiguration()
            self.cameraConfiguration = .failed
            return
        }
        
        guard addVideoDataOutput() else {
            self.session.commitConfiguration()
            self.cameraConfiguration = .failed
            return
        }
        
        session.commitConfiguration()
        self.cameraConfiguration = .success
    }
    
    /**This method tries to an AVCaptureDeviceInput to the current AVCaptureSession.*/
    private func addVideoDeviceInput() -> Bool {
        
        /// Default : Get back camara
        guard let camera  = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return false
        }
        
        do {
            let videoDeviceInput = try AVCaptureDeviceInput(device: camera)
            if session.canAddInput(videoDeviceInput) {
                session.addInput(videoDeviceInput)
                return true
            }
            else {
                return false
            }
        }
        catch {
            fatalError("Cannot create video device input")
        }
    }
    
    /**This method tries to an AVCaptureVideoDataOutput to the current AVCaptureSession.*/
    private func addVideoDataOutput() -> Bool {
        
        let sampleBufferQueue = DispatchQueue(label: "sampleBufferQueue")
        videoDataOutput.setSampleBufferDelegate(self, queue: sampleBufferQueue)
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutput.videoSettings = [ String(kCVPixelBufferPixelFormatTypeKey) : kCMPixelFormat_32BGRA]
        
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            videoDataOutput.connection(with: .video)?.videoOrientation = .portrait
            return true
        }
        return false
    }
}


/**
 Delegate the CVPixelBuffer of the frame seen by the camera currently.
 */
extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // Converts the CMSampleBuffer to a CVPixelBuffer.
        let pixelBuffer: CVPixelBuffer? = CMSampleBufferGetImageBuffer(sampleBuffer)
        
        guard let imagePixelBuffer = pixelBuffer else {
            return
        }
        
        // Delegates the pixel buffer to the ViewController.
        delegate?.didOutput(pixelBuffer: imagePixelBuffer)
    }
}
