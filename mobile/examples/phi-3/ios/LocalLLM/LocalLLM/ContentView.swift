// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import SwiftUI


struct Message: Identifiable {
    let id = UUID()
    var text: String
    let isUser: Bool
}

struct ContentView: View {
    @State private var userInput: String = ""
    @State private var messages: [Message] = []  // Store chat messages locally
    @State private var isGenerating: Bool = false  // Track token generation state
    @State private var stats: String = ""  // token generation stats
    @State private var showAlert: Bool = false
    @State private var errorMessage: String = ""

    private let generator = GenAIGenerator()
    
    var body: some View {
        VStack {
            // ChatBubbles
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    ForEach(messages) { message in
                        ChatBubble(text: message.text, isUser: message.isUser)
                            .padding(.horizontal, 20)
                    }
                    if !stats.isEmpty {
                        Text(stats)
                            .font(.footnote)
                            .foregroundColor(.gray)
                            .padding(.horizontal, 20)
                            .padding(.top, 5)
                            .multilineTextAlignment(.center)
                    }
                }
                .padding(.top, 20)
            }

            
            // User input 
            HStack {
                TextField("Type your message...", text: $userInput)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(20)
                    .padding(.horizontal)
                
                Button(action: {
                    // Check for non-empty input
                    guard !userInput.trimmingCharacters(in: .whitespaces).isEmpty else { return }
                    
                    messages.append(Message(text: userInput, isUser: true))
                    messages.append(Message(text: "", isUser: false))  // Placeholder for AI response
                    

                    // clear previously generated tokens
                    SharedTokenUpdater.shared.clearTokens()

                    let prompt = userInput
                    userInput = ""
                    isGenerating = true
                
        
                    DispatchQueue.global(qos: .background).async {
                        generator.generate(prompt)
                    }
                }) {
                    Image(systemName: "paperplane.fill")
                        .foregroundColor(.white)
                        .padding()
                        .background(isGenerating ? Color.gray : Color.pastelGreen)
                        .clipShape(Circle())
                        .padding(.trailing, 10)
                }
                .disabled(isGenerating)
            }
            .padding(.bottom, 20)
        }
        .background(Color(.systemGroupedBackground))
        .edgesIgnoringSafeArea(.bottom)
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("TokenGenerationCompleted"))) { _ in
            isGenerating = false  // Re-enable the button when token generation is complete
        }
        .onReceive(SharedTokenUpdater.shared.$decodedTokens) { tokens in
            // update model response
            if let lastIndex = messages.lastIndex(where: { !$0.isUser }) {
                let combinedText = tokens.joined(separator: "")
                messages[lastIndex].text = combinedText
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("TokenGenerationStats"))) { notification in
            if let userInfo = notification.userInfo,
               let promptProcRate = userInfo["promptProcRate"] as? Double,
               let tokenGenRate = userInfo["tokenGenRate"] as? Double {
                stats = String(format: "Token generation rate: %.2f tokens/s. Prompt processing rate: %.2f tokens/s", tokenGenRate, promptProcRate)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSNotification.Name("TokenGenerationError"))) { notification in
            if let userInfo = notification.userInfo, let error = userInfo["error"] as? String {
                    errorMessage = error
                    isGenerating = false
                    showAlert = true
            }
        }
        .alert(isPresented: $showAlert) {
            Alert(
                title: Text("Error"),
                message: Text(errorMessage),
                dismissButton: .default(Text("OK"))
            )
        }
        
    }
}

struct ChatBubble: View {
    var text: String
    var isUser: Bool

    var body: some View {
        HStack {
            if isUser {
                Spacer()
                Text(text)
                    .padding()
                    .background(Color.pastelGreen)
                    .foregroundColor(.white)
                    .cornerRadius(25)
                    .padding(.horizontal, 10)
            } else {
                Text(text)
                    .padding()
                    .background(Color(.systemGray5))
                    .foregroundColor(.black)
                    .cornerRadius(25)
                    .padding(.horizontal, 10)
                Spacer()
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

// Extension for a pastel green color
extension Color {
    static let pastelGreen = Color(red: 0.6, green: 0.9, blue: 0.6)
}
