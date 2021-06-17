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

// MARK: InferenceViewControllerDelegate Method Declarations

protocol InferenceViewControllerDelegate {
    func didChangeThreadCount(to count: Int32)
}

class InferenceViewController: UIViewController {
    // MARK: Storyboard Outlets

    @IBOutlet var tableView: UITableView!
    @IBOutlet var threadStepper: UIStepper!
    @IBOutlet var stepperValueLabel: UILabel!
    
    // MARK: Inference related display results and info

    private enum InferenceResults: Int, CaseIterable {
        case InferenceInfo
    }
    
    private enum InferenceInfo: Int, CaseIterable {
        case Resolution
        case Crop
        case InferenceTime
        
        func displayString() -> String {
            var toReturn = ""
            
            switch self {
            case .Resolution:
                toReturn = "Resolution"
            case .Crop:
                toReturn = "Crop"
            case .InferenceTime:
                toReturn = "Inference Time"
            }
            return toReturn
        }
    }
    
    var inferenceTime: Double = 0
    var wantedInputWidth: Int = 0
    var wantedInputHeight: Int = 0
    var resolution = CGSize.zero
    var threadCountLimit: Int = 0
    var currentThreadCount: Int32 = 0
    private let minThreadCount = 1
    
    var delegate: InferenceViewControllerDelegate?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Set up stepper
        threadStepper.isUserInteractionEnabled = true
        threadStepper.maximumValue = Double(threadCountLimit)
        threadStepper.minimumValue = Double(minThreadCount)
        threadStepper.value = Double(currentThreadCount)
    }
    
    // Delegate the change of number of threads to View Controller and change the stepper display
    @IBAction func onClickThreadStepper(_ sender: Any) {
        delegate?.didChangeThreadCount(to: Int32(threadStepper.value))
        currentThreadCount = Int32(threadStepper.value)
        stepperValueLabel.text = "\(currentThreadCount)"
    }
}

// MARK: UITableView Data Source

extension InferenceViewController: UITableViewDelegate, UITableViewDataSource {
    func numberOfSections(in tableView: UITableView) -> Int {
        return InferenceResults.allCases.count
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        guard let inferenceResults = InferenceResults(rawValue: section) else {
            return 0
        }
        
        var rowCount = 0
        switch inferenceResults {
        case .InferenceInfo:
            rowCount = InferenceInfo.allCases.count
        }

        return rowCount
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "INFO_CELL") as! InfoCell
        
        guard let inferenceResults = InferenceResults(rawValue: indexPath.section) else {
            return cell
        }
        
        var fieldName = ""
        var info = ""
        
        switch inferenceResults {
        case .InferenceInfo:
            let tuple = displayStringsForInferenceInfo(atRow: indexPath.row)
            fieldName = tuple.0
            info = tuple.1
        }
        cell.fieldNameLabel.font = UIFont.systemFont(ofSize: 14.0, weight: .regular)
        cell.fieldNameLabel.textColor = UIColor.black
        cell.fieldNameLabel.text = fieldName
        cell.infoLabel.text = info

        return cell
    }
    
    // This method formats the display of additional information related to the inferences.
    func displayStringsForInferenceInfo(atRow row: Int) -> (String, String) {
        var fieldName: String = ""
        var info: String = ""
        
        guard let inferenceInfo = InferenceInfo(rawValue: row) else {
            return (fieldName, info)
        }
        
        fieldName = inferenceInfo.displayString()
        
        switch inferenceInfo {
        case .Resolution:
            info = "\(Int(resolution.width))x\(Int(resolution.height))"
        case .Crop:
            info = "\(wantedInputWidth)x\(wantedInputHeight)"
        case .InferenceTime:
            info = String(format: "%.2fms", inferenceTime)
        }
        
        return (fieldName, info)
    }
}
