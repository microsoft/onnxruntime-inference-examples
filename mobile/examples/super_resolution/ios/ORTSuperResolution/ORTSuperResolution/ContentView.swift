// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


import SwiftUI

struct ContentView: View {
    @State private var performSuperRes = false
    
    // TODO: modify the output result (String) to be an image to get displayed on screen
    func runOrtSuperResolution() -> String {
        do {
            try ORTSuperResolutionPerformer.performSuperResolution()
            return "Ok"
        } catch let error as NSError {
            return "Error: \(error.localizedDescription)"
        }
    }
    
    var body: some View {
        VStack {
            VStack {
                Text("ORTSuperResolution").font(.title).bold()
                    .frame(width: 400, height: 80)
                    .border(Color.purple, width: 4)
                    .background(Color.purple)
                
                Text("Input low resolution image: ").frame(width: 380, height: 40, alignment:.leading)
                
                Image("test_superresolution")
                
                Button("Perform Super Resolution") {
                  performSuperRes.toggle()
                }

                if performSuperRes {
                    Text("Output high resolution image: ").frame(width: 380, height: 40, alignment:.leading)
                    // TODO: call run OrtSuperResolution and display Image
                    Text("Ort Super Resolution Result: \(runOrtSuperResolution())")
                }
                Spacer()
                
            }
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
