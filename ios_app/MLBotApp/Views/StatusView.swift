import SwiftUI

struct StatusView: View {
    @ObservedObject var client: APIClient
    @State private var status: StatusResponse?
    @State private var errorMessage: String?
    @State private var loading = false

    var body: some View {
        NavigationStack {
            Group {
                if loading && status == nil {
                    ProgressView("Загрузка…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let err = errorMessage {
                    VStack(spacing: 12) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.largeTitle)
                            .foregroundStyle(.orange)
                        Text(err)
                            .multilineTextAlignment(.center)
                            .padding()
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let s = status {
                    statusContent(s)
                } else {
                    Text("Нажмите «Обновить» или настройте API в Настройках")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .navigationTitle("Статус")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        Task { await load() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(loading || !client.isConfigured)
                }
            }
            .refreshable {
                await load()
            }
            .task {
                if status == nil && client.isConfigured {
                    await load()
                }
            }
        }
    }

    private func statusContent(_ s: StatusResponse) -> some View {
        List {
            Section("Бот") {
                HStack {
                    Circle()
                        .fill(s.isRunning ? Color.green : Color.red)
                        .frame(width: 12, height: 12)
                    Text(s.isRunning ? "Работает" : "Остановлен")
                        .fontWeight(.medium)
                }
            }
            Section("Счёт") {
                LabeledContent("Баланс", value: "$\(s.walletBalance, specifier: "%.2f")")
                LabeledContent("Доступно", value: "$\(s.availableBalance, specifier: "%.2f")")
                LabeledContent("В позициях", value: "$\(s.totalMargin, specifier: "%.2f")")
            }
            Section("Итого") {
                LabeledContent("PnL", value: "\(s.totalPnl >= 0 ? "+" : "")$\(s.totalPnl, specifier: "%.2f")")
                LabeledContent("Винрейт", value: "\(s.winRate, specifier: "%.1f")%")
                LabeledContent("Сделок", value: "\(s.totalTrades)")
            }
            if !s.positions.isEmpty {
                Section("Позиции") {
                    ForEach(s.positions, id: \.symbol) { pos in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(pos.symbol)
                                    .fontWeight(.semibold)
                                Spacer()
                                Text(pos.side)
                                    .foregroundStyle(pos.side == "Buy" ? .green : .red)
                            }
                            HStack {
                                Text("Вход: $\(pos.entry, specifier: "%.2f")")
                                Text("Тек: $\(pos.current, specifier: "%.2f")")
                            }
                            .font(.caption)
                            Text("PnL: \(pos.pnl >= 0 ? "+" : "")$\(pos.pnl, specifier: "%.2f") (\(pos.pnlPct, specifier: "%.2f")%)")
                                .font(.caption)
                                .foregroundStyle(pos.pnl >= 0 ? .green : .red)
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
            if !s.activeSymbols.isEmpty {
                Section("Активные пары") {
                    Text(s.activeSymbols.joined(separator: ", "))
                        .font(.caption)
                }
            }
        }
    }

    private func load() async {
        guard client.isConfigured else {
            errorMessage = APIError.noCredentials.errorDescription
            return
        }
        loading = true
        errorMessage = nil
        defer { loading = false }
        do {
            status = try await client.status()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
