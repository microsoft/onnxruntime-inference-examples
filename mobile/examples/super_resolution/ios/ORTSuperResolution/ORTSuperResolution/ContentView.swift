// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


import SwiftUI

struct ContentView: View {
    @State private var performSuperRes = false
    
    func runOrtSuperResolution() -> UIImage? {
        do {
            let outputImage = try ORTSuperResolutionPerformer.performSuperResolution()
            return outputImage
        } catch let error as NSError {
            print("Error: \(error.localizedDescription)")
            return nil
        }
    }
    
    var body: some View {
        ScrollView {
            VStack {
                VStack {
                    Text("ORTSuperResolution").font(.title).bold()
                        .frame(width: 400, height: 80)
                        .border(Color.purple, width: 4)
                        .background(Color.purple)
                    
                    Text("Low resolution image: ").frame(width: 350, height: 40, alignment:.leading)
                    
                    Image("cat_224x224").resizable().aspectRatio(1, contentMode: .fit).frame(width: 250, height: 250)
                    
                    Button("Perform Super Resolution") {
                        performSuperRes.toggle()
                    }
                    
                    if performSuperRes {
                        Text("Higher resolution image: ").frame(width: 350, height: 40, alignment:.leading)
                        
                        if let outputImage = runOrtSuperResolution() {
                            Image(uiImage: outputImage).resizable()
                                .aspectRatio(1, contentMode: .fit).frame(width: 250, height: 250)
                                
                        } else {
                            Text("Unable to perform super resolution. ").frame(width: 350, height: 40, alignment:.leading)
                        }
                    }
                    Spacer()
                }
            }
            .padding()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
