import SwiftUI
import Charts

struct AnalyticsView: View {
    @ObservedObject var client: APIClient
    @State private var equity: EquityCurveResponse?
    @State private var errorMessage: String?
    @State private var loading = false

    var body: some View {
        Group {
            if loading && equity == nil {
                ProgressView("Загрузка…")
            } else if let err = errorMessage {
                Text(err).foregroundStyle(.red).padding()
            } else if let eq = equity, !eq.points.isEmpty {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Кривая капитала (PnL)")
                            .font(.headline)
                        Chart(Array(eq.points.enumerated()), id: \.offset) { i, p in
                            LineMark(
                                x: .value("Шаг", i),
                                y: .value("PnL", p.pnlCum)
                            )
                            .foregroundStyle(eq.points.last?.pnlCum ?? 0 >= 0 ? Color.green : Color.red)
                        }
                        .frame(height: 220)
                        Text("Итого PnL: \(eq.totalPnl >= 0 ? "+" : "")$\(eq.totalPnl, specifier: "%.2f")")
                            .font(.title2)
                    }
                    .padding()
                }
            } else {
                Text("Нет данных по сделкам для графика").foregroundStyle(.secondary).padding()
            }
        }
        .navigationTitle("Аналитика")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: { Image(systemName: "arrow.clockwise") }
                .disabled(loading || !client.isConfigured)
            }
        }
        .task { if equity == nil && client.isConfigured { await load() } }
    }

    private func load() async {
        guard client.isConfigured else { errorMessage = "Настройте API"; return }
        loading = true
        errorMessage = nil
        defer { loading = false }
        do {
            equity = try await client.equityCurve()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
