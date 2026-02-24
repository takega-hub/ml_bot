import SwiftUI

struct ControlView: View {
    @ObservedObject var client: APIClient
    @State private var loading = false
    @State private var message: String?
    @State private var isRunning: Bool?

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                if let running = isRunning {
                    HStack(spacing: 12) {
                        Circle()
                            .fill(running ? Color.green : Color.red)
                            .frame(width: 16, height: 16)
                        Text(running ? "Бот запущен" : "Бот остановлен")
                            .font(.headline)
                    }
                }
                if let msg = message {
                    Text(msg)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                HStack(spacing: 20) {
                    Button {
                        Task { await doStart() }
                    } label: {
                        Label("Старт", systemImage: "play.fill")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green.opacity(0.2))
                            .foregroundStyle(.green)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .disabled(loading || !client.isConfigured)
                    Button {
                        Task { await doStop() }
                    } label: {
                        Label("Стоп", systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.red.opacity(0.2))
                            .foregroundStyle(.red)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .disabled(loading || !client.isConfigured)
                }
                .padding(.horizontal)
                Spacer()
            }
            .padding(.top, 32)
            .navigationTitle("Управление")
            .task {
                if client.isConfigured && isRunning == nil {
                    await refreshState()
                }
            }
        }
    }

    private func doStart() async {
        guard client.isConfigured else { return }
        loading = true
        message = nil
        defer { loading = false }
        do {
            _ = try await client.start()
            isRunning = true
            message = "Бот запущен"
        } catch {
            message = error.localizedDescription
        }
    }

    private func doStop() async {
        guard client.isConfigured else { return }
        loading = true
        message = nil
        defer { loading = false }
        do {
            _ = try await client.stop()
            isRunning = false
            message = "Бот остановлен"
        } catch {
            message = error.localizedDescription
        }
    }

    private func refreshState() async {
        guard client.isConfigured else { return }
        do {
            let s = try await client.status()
            isRunning = s.isRunning
        } catch {
            message = error.localizedDescription
        }
    }
}
