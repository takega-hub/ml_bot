import SwiftUI

struct EmergencyView: View {
    @ObservedObject var client: APIClient
    @State private var loading = false
    @State private var message: String?
    @State private var showConfirm = false

    var body: some View {
        VStack(spacing: 24) {
            Text("Экстренные действия необратимы. Используйте только при необходимости.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding()
            Button {
                showConfirm = true
            } label: {
                Label("Стоп и закрыть все позиции", systemImage: "exclamationmark.triangle.fill")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red.opacity(0.3))
                    .foregroundStyle(.red)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
            .disabled(loading || !client.isConfigured)
            if let msg = message {
                Text(msg)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding()
            }
            Spacer()
        }
        .navigationTitle("Экстренные")
        .confirmationDialog("Подтверждение", isPresented: $showConfirm) {
            Button("Стоп и закрыть все", role: .destructive) {
                Task { await emergencyStop() }
            }
            Button("Отмена", role: .cancel) {}
        } message: {
            Text("Остановить бота и закрыть все открытые позиции по рынку?")
        }
    }

    private func emergencyStop() async {
        guard client.isConfigured else { return }
        loading = true
        message = nil
        defer { loading = false }
        do {
            let r = try await client.emergencyStopAll()
            message = r.message ?? "Выполнено. Закрыто: \(r.closedPositions.joined(separator: ", "))"
        } catch {
            message = error.localizedDescription
        }
    }
}
