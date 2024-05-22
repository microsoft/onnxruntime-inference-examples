// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import SwiftUI

struct ContentView: View {
    @ObservedObject var tokenUpdater = SharedTokenUpdater.shared
    
    var body: some View {
        VStack {
            ScrollView {
                VStack(alignment: .leading) {
                    ForEach(tokenUpdater.decodedTokens, id: \.self) { token in
                        Text(token)
                           .padding(.horizontal, 5)
                    }
                }
                .padding()
            }
            Button("Generate Tokens") {
                DispatchQueue.global(qos: .background).async {
                    // TODO: add user prompt question UI
                    GenAIGenerator.generate("Who is the current US president?");
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
