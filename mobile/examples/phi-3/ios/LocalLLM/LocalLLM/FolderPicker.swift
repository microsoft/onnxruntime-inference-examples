// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import SwiftUI
import UIKit

struct FolderPicker: UIViewControllerRepresentable {
  var onPick: (URL?) -> Void

  func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
    let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
    picker.allowsMultipleSelection = false
    picker.delegate = context.coordinator
    return picker
  }

  func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

  func makeCoordinator() -> Coordinator {
    Coordinator(onPick: onPick)
  }

  class Coordinator: NSObject, UIDocumentPickerDelegate {
    let onPick: (URL?) -> Void

    init(onPick: @escaping (URL?) -> Void) {
      self.onPick = onPick
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
      onPick(urls.first)
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
      onPick(nil)
    }
  }
}
