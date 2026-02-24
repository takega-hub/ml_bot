import SwiftUI

struct StatsView: View {
    @ObservedObject var client: APIClient
    @State private var stats: StatsResponse?
    @State private var errorMessage: String?
    @State private var loading = false

    var body: some View {
        Group {
            if loading && stats == nil {
                ProgressView("Загрузка…")
            } else if let err = errorMessage {
                Text(err).foregroundStyle(.red).padding()
            } else if let s = stats {
                List {
                    Section("Итого") {
                        LabeledContent("Общий PnL", value: "\(s.totalPnl >= 0 ? "+" : "")$\(s.totalPnl, specifier: "%.2f")")
                        LabeledContent("Винрейт", value: "\(s.winRate, specifier: "%.1f")%")
                        LabeledContent("Всего сделок", value: "\(s.totalTrades)")
                        LabeledContent("Закрыто", value: "\(s.closedCount)")
                        LabeledContent("Открыто", value: "\(s.openCount)")
                    }
                    Section("Результаты") {
                        LabeledContent("Прибыльных", value: "\(s.winsCount)")
                        LabeledContent("Убыточных", value: "\(s.lossesCount)")
                        if let avg = s.avgWin {
                            LabeledContent("Средний выигрыш", value: "$\(avg, specifier: "%.2f")")
                        }
                        if let avg = s.avgLoss {
                            LabeledContent("Средний проигрыш", value: "$\(avg, specifier: "%.2f")")
                        }
                    }
                }
            } else {
                Text("Нажмите «Обновить»").foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Статистика")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: { Image(systemName: "arrow.clockwise") }
                .disabled(loading || !client.isConfigured)
            }
        }
        .refreshable { await load() }
        .task { if stats == nil && client.isConfigured { await load() } }
    }

    private func load() async {
        guard client.isConfigured else { errorMessage = "Настройте API"; return }
        loading = true
        errorMessage = nil
        defer { loading = false }
        do {
            stats = try await client.stats()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
