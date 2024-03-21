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

// create sub class of UICollectionViewCell for images
class Mycell : UICollectionViewCell {
     let imageView = UIImageView()
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        addSubview(imageView)
    }
    
    required init(coder: NSCoder) {
        fatalError()
    }
    
    override func layoutSubviews() {
        super.layoutSubviews()
        imageView.frame = bounds
    }
}


class ViewController: UIViewController , PHPickerViewControllerDelegate, UICollectionViewDataSource {
    
    // store selected images
    private var selectedImages = [String: PHPickerResult]()
    
    // Handle all model and data preprocessing and run inference
    private var modelHandler: ModelHandler? = ModelHandler(
        modelFileInfo: (name: "yolov8n", extension: "ort"),
        labelsFileInfo: (name: "labelmap", extension: "txt"))
    
    // create collection view to store selected images
    private let collectionView: UICollectionView = {
        // create layout for UI collection
       let layout = UICollectionViewFlowLayout()
        layout.itemSize = CGSize(width: 350, height: 400)
        let collectionView = UICollectionView(frame: .zero, collectionViewLayout: layout)
        collectionView.backgroundColor = .yellow
        collectionView.register(Mycell.self, forCellWithReuseIdentifier: "cell")
        return collectionView
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // add UI collection view
        view.addSubview(collectionView)
        collectionView.dataSource = self
        collectionView.frame = view.bounds
        
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
        let vc = PHPickerViewController(configuration: config)
        vc.delegate = self
        present(vc, animated: true)
    }
    
    // call image picker method
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        
        // create group to select multiple images
        let group = DispatchGroup()
        
        // store previous selection
        let existingSelection = self.selectedImages
    
        var newSelection = [String: PHPickerResult]()
        
        //  this will print result data
        results.forEach { result in
            group.enter()
            result.itemProvider.loadObject(ofClass: UIImage.self)  { [weak self] reading, error in
                defer {
                    group.leave()
                }
                // store image identifier
                let identifier = result.assetIdentifier!
                
                newSelection[identifier] = existingSelection[identifier] ?? result
        
                // store image
                guard let image = reading as? UIImage, error == nil else{
                    return
                }
              
                // append image to existing selected images
                self?.images.append(image)
            }
        }
        
        // after selecting images show images to the user
        group.notify(queue: .main){
            self.collectionView.reloadData()
            // Track the selection in case the user deselects it later.
            self.selectedImages = newSelection
            
            // start image processing
            self.startImageProcessing()
        }
    }
    
    
    // start image processing on all images selected by user
    func startImageProcessing(){
        // loop over all images
        for result in selectedImages {
            
            // store image identifierx
            let identifier = result.value.assetIdentifier!
            
            // process image using it's indetifier
            processSingleImageUsingIndentifier(assetIdentifier: identifier)
        }
    }
    
    // process image
    func processSingleImageUsingIndentifier(assetIdentifier: String){
        // item provider which is obtain from image picker result
        let itemProvider = selectedImages[assetIdentifier]!.itemProvider
        
        // access image data using item provider method
        if itemProvider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            itemProvider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, error in
                
             // image data (buffer data)
                guard let data = data,
                     // create cgImage from buffer data
                      let cgImageSource = CGImageSourceCreateWithData(data as CFData, nil) else { return }
                
                     //  Convert CgImageSource to CGImage
                      guard  let cgImage = CGImageSourceCreateImageAtIndex(cgImageSource, 0, nil)  else { return }
                    
                    // run model
                    self.modelHandler?.runModel(cgImage: cgImage)
            }
        }
    }
    
    private var images = [UIImage]()
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return images.count
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        guard let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "cell", for: indexPath) as? Mycell else {fatalError()}
         cell.imageView.image = images[indexPath.row]
        return cell
    }
    
}

