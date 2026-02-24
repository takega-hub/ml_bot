import SwiftUI

struct DashboardView: View {
    @ObservedObject var client: APIClient
    @State private var dashboard: DashboardResponse?
    @State private var errorMessage: String?
    @State private var loading = false

    var body: some View {
        NavigationStack {
            Group {
                if loading && dashboard == nil {
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
                } else if let d = dashboard {
                    dashboardContent(d)
                } else {
                    Text("Нажмите «Обновить» или настройте API в Настройках")
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .navigationTitle("Дашборд")
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
                if dashboard == nil && client.isConfigured {
                    await load()
                }
            }
        }
    }

    private func dashboardContent(_ d: DashboardResponse) -> some View {
        List {
            Section("Баланс") {
                LabeledContent("Текущий", value: "$\(d.walletBalance, specifier: "%.2f")")
                LabeledContent("Доступно", value: "$\(d.availableBalance, specifier: "%.2f")")
                LabeledContent("В позициях", value: "$\(d.totalMargin, specifier: "%.2f")")
                if d.walletBalance > 0 {
                    LabeledContent("PnL %", value: "\(d.totalPnlPct >= 0 ? "+" : "")\(d.totalPnlPct, specifier: "%.2f")%")
                }
            }
            Section("Позиции") {
                LabeledContent("Открыто", value: "\(d.openPositionsCount)")
                if d.openPositionsCount > 0 {
                    LabeledContent("Текущий PnL", value: "\(d.openPositionsPnl >= 0 ? "+" : "")$\(d.openPositionsPnl, specifier: "%.2f")")
                }
            }
            Section("Сегодня") {
                LabeledContent("Сделок", value: "\(d.todayTrades)")
                LabeledContent("PnL", value: "\(d.todayPnl >= 0 ? "+" : "")$\(d.todayPnl, specifier: "%.2f")")
            }
            Section("Неделя") {
                LabeledContent("PnL", value: "\(d.weekPnl >= 0 ? "+" : "")$\(d.weekPnl, specifier: "%.2f")")
                LabeledContent("Винрейт", value: "\(d.weekWinrate, specifier: "%.1f")%")
                LabeledContent("Сделок", value: "\(d.weekTrades)")
            }
            Section("Система") {
                HStack {
                    Circle()
                        .fill(d.isRunning ? Color.green : Color.red)
                        .frame(width: 10, height: 10)
                    Text(d.isRunning ? "Бот работает" : "Бот остановлен")
                }
                LabeledContent("Активных пар", value: "\(d.activeSymbolsCount)")
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
            dashboard = try await client.dashboard()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
