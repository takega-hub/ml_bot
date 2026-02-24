import SwiftUI

struct RiskSettingsView: View {
    @ObservedObject var client: APIClient
    @State private var risk: RiskResponse?
    @State private var errorMessage: String?
    @State private var loading = false
    @State private var message: String?

    var body: some View {
        Group {
            if loading && risk == nil {
                ProgressView("Загрузка…")
            } else if let err = errorMessage {
                Text(err).foregroundStyle(.red).padding()
            } else if let r = risk {
                List {
                    Section("Позиция") {
                        if let v = r.marginPctBalance {
                            LabeledContent("Маржа % от баланса", value: "\(v * 100, specifier: "%.0f")%")
                        }
                        if let v = r.baseOrderUsd {
                            LabeledContent("Базовая сумма $", value: "\(v, specifier: "%.2f")")
                        }
                    }
                    Section("Стопы") {
                        if let v = r.stopLossPct {
                            LabeledContent("Stop Loss %", value: "\(v * 100, specifier: "%.2f")%")
                        }
                        if let v = r.takeProfitPct {
                            LabeledContent("Take Profit %", value: "\(v * 100, specifier: "%.2f")%")
                        }
                    }
                    Section("Трейлинг") {
                        if let v = r.enableTrailingStop {
                            LabeledContent("Включен", value: v ? "Да" : "Нет")
                        }
                        if let v = r.trailingStopActivationPct {
                            LabeledContent("Активация %", value: "\(v * 100, specifier: "%.2f")%")
                        }
                        if let v = r.trailingStopDistancePct {
                            LabeledContent("Расстояние %", value: "\(v * 100, specifier: "%.2f")%")
                        }
                    }
                    Section("Прочее") {
                        if let v = r.enablePartialClose {
                            LabeledContent("Частичное закрытие", value: v ? "Да" : "Нет")
                        }
                        if let v = r.enableBreakeven {
                            LabeledContent("Безубыток", value: v ? "Да" : "Нет")
                        }
                        if let v = r.enableLossCooldown {
                            LabeledContent("Cooldown после убытков", value: v ? "Да" : "Нет")
                        }
                    }
                    if let msg = message {
                        Section {
                            Text(msg).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                .navigationBarTitleDisplayMode(.inline)
            } else {
                Text("Обновите или настройте API").foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Настройки риска")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: { Image(systemName: "arrow.clockwise") }
                .disabled(loading || !client.isConfigured)
            }
        }
        .task { if risk == nil && client.isConfigured { await load() } }
    }

    private func load() async {
        guard client.isConfigured else { errorMessage = "Настройте API"; return }
        loading = true
        errorMessage = nil
        defer { loading = false }
        do {
            risk = try await client.risk()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
