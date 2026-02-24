import SwiftUI

struct SettingsView: View {
    @ObservedObject var client: APIClient
    @State private var baseURLInput: String = ""
    @State private var apiKeyInput: String = ""
    @State private var saved = false
    @State private var testMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("URL сервера", text: $baseURLInput)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    Text("Например: http://192.168.1.100:8765 или https://your-ngrok.io")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } header: {
                    Text("Сервер")
                }
                Section {
                    SecureField("API ключ", text: $apiKeyInput)
                } header: {
                    Text("Ключ")
                } footer: {
                    Text("Значение MOBILE_API_KEY из .env на сервере бота (или ваш Telegram ID).")
                }
                Section {
                    Button {
                        save()
                    } label: {
                        HStack {
                            Text("Сохранить")
                            Spacer()
                            if saved {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(.green)
                            }
                        }
                    }
                    if client.isConfigured {
                        Button {
                            Task { await testConnection() }
                        } label: {
                            HStack {
                                Text("Проверить связь")
                                Spacer()
                                if let msg = testMessage {
                                    Text(msg)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Настройки")
            .onAppear {
                baseURLInput = client.baseURL
                apiKeyInput = client.apiKey
            }
        }
    }

    private func save() {
        client.baseURL = baseURLInput.trimmingCharacters(in: .whitespacesAndNewlines)
        client.apiKey = apiKeyInput
        saved = true
        testMessage = nil
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            saved = false
        }
    }

    private func testConnection() async {
        testMessage = "…"
        do {
            _ = try await client.health()
            testMessage = "OK"
        } catch {
            testMessage = "Ошибка"
        }
    }
}
