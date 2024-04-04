//
//  ViewController.swift
//  yolo-ios
//
//  Created by Jayashree Patil on 29/02/24.
//

import UIKit
import PhotosUI
import AVKit
import CoreImage
import Accelerate
import Accelerate.vImage
import onnxruntime_objc

private var session: ORTSession?
private var env: ORTEnv?

class ViewController: UIViewController , PHPickerViewControllerDelegate {
    @IBOutlet weak var spinner: UIActivityIndicatorView!
    @IBOutlet weak var start: UIButton!
    @IBOutlet weak var errorText: UILabel!
    @IBOutlet weak var uiImageView: UIImageView!
    
    private var selectedImage: UIImage!
    private var selectedImageAsset: PHPickerResult!
    private var selectedImageAssetIdentifier: String!
    
    // Handle all model and data preprocessing and run inference
    private var modelHandler: ModelHandler? = ModelHandler(
        modelFileInfo: (name: "yolov8n", extension: "ort"),
        labelsFileInfo: (name: "labelmap", extension: "txt"))
    
    // view did load
    override func viewDidLoad() {
        super.viewDidLoad()
        // spinner view
        spinner.hidesWhenStopped = true
        spinner.style = .large
        spinner.color = .darkGray
      
        // Do any additional setup after loading the view.
        title = "Object Detection"
        // adding add image "+" button
       navigationItem.rightBarButtonItem = UIBarButtonItem(barButtonSystemItem: .add , target: self, action: #selector(didTapAdd))
    }
    
    // config PHPicker on add image ("+" icon onPress)
    @objc private func didTapAdd(){
        var config = PHPickerConfiguration(photoLibrary: .shared())
        config.selectionLimit = 1
        config.filter = .images
        let phViewController = PHPickerViewController(configuration: config)
        phViewController.delegate = self
        present(phViewController, animated: true)
    }
    
    // call image picker method
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        errorText.text = " "
        picker.dismiss(animated: true)
        
        // create group to select images
        let group = DispatchGroup()
        
       // used to store image result selected by PHPicker
        var phImageResult : PHPickerResult!
        
        //  this will print result data
        results.forEach { result in
            group.enter()
            result.itemProvider.loadObject(ofClass: UIImage.self)  { [weak self] reading, error in
                defer {
                    group.leave()
                }
               
                // assign selected image asset indentifier
                self?.selectedImageAssetIdentifier =  result.assetIdentifier!
                
                // store PH image result
                phImageResult =  result
        
                // store image
                guard let image = reading as? UIImage, error == nil else{
                    return
                }
                
                // assign image selected by user
                self?.selectedImage = image
            }
        }
        
        // after selecting images show images to the user
        group.notify(queue: .main){
            //  assign selected image ph result
            self.selectedImageAsset = phImageResult
            
            if self.selectedImageAsset == nil {
             return
            } else {
                // display selected image
                self.selectedImage.prepareForDisplay { [weak self] preparedImage in
                        DispatchQueue.main.async {
                            self?.uiImageView.image = preparedImage
                        }
                    }
            }
        }
    }
    

    // handle "Start Processing" button click
    @IBAction func startProcessing(_ sender: UIButton) {
        spinner.startAnimating()
        self.processSingleImageUsingIndentifier(assetIdentifier: self.selectedImageAssetIdentifier)
    }
    
    // process image
    func processSingleImageUsingIndentifier(assetIdentifier: String){
        let itemProvider = self.selectedImageAsset.itemProvider
        
        // access image data using item provider method
        if itemProvider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            itemProvider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, error in
                
             // image data (buffer data)
                guard let data = data,
                     // create cgImage from buffer data
                      let cgImageSource = CGImageSourceCreateWithData(data as CFData, nil) else { return}
             
                     //  Convert CgImageSource to CGImage
                      guard  let cgImage = CGImageSourceCreateImageAtIndex(cgImageSource, 0, nil)  else { return }
                       
                        // run model
                     let processedImage =  self.modelHandler?.runModel(cgImage: cgImage)
                
                    // show processed image to the user
                    self.stopLoaderAndShowProcessedImage(image: processedImage)
            }
        }
    }
    
    // show processed image to the user
    func stopLoaderAndShowProcessedImage(image: UIImage?) {
        if(image == nil){
            DispatchQueue.main.async {
                self.spinner.stopAnimating();
                self.errorText.text = "Error accoured while processing"
            }
        }else{
            image?.prepareForDisplay { [weak self] preparedImage in
                        DispatchQueue.main.async {
                            self?.spinner.stopAnimating();
                            self?.uiImageView.image = preparedImage
                        }
              }
        }
    }
}

