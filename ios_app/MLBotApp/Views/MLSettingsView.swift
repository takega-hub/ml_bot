import SwiftUI

struct MLSettingsView: View {
    @ObservedObject var client: APIClient
    @State private var ml: MLResponse?
    @State private var errorMessage: String?
    @State private var loading = false

    var body: some View {
        Group {
            if loading && ml == nil {
                ProgressView("Загрузка…")
            } else if let err = errorMessage {
                Text(err).foregroundStyle(.red).padding()
            } else if let m = ml {
                List {
                    Section("MTF стратегия") {
                        if let v = m.useMtfStrategy {
                            LabeledContent("Включена", value: v ? "Да" : "Нет")
                        }
                        if let v = m.mtfConfidenceThreshold1h {
                            LabeledContent("Порог 1h %", value: "\(v * 100, specifier: "%.0f")%")
                        }
                        if let v = m.mtfConfidenceThreshold15m {
                            LabeledContent("Порог 15m %", value: "\(v * 100, specifier: "%.0f")%")
                        }
                        if let v = m.mtfAlignmentMode {
                            LabeledContent("Режим", value: v)
                        }
                    }
                    Section("Уверенность") {
                        if let v = m.confidenceThreshold {
                            LabeledContent("Модели %", value: "\(v * 100, specifier: "%.0f")%")
                        }
                        if let v = m.minConfidenceForTrade {
                            LabeledContent("Для сделки %", value: "\(v * 100, specifier: "%.0f")%")
                        }
                    }
                    Section("Прочее") {
                        if let v = m.atrFilterEnabled {
                            LabeledContent("Фильтр ATR", value: v ? "Вкл" : "Выкл")
                        }
                        if let v = m.autoOptimizeStrategies {
                            LabeledContent("Автообновление", value: v ? "Вкл" : "Выкл")
                        }
                        if let v = m.useFixedSlFromRisk {
                            LabeledContent("SL из риска", value: v ? "Да" : "Нет")
                        }
                    }
                }
            } else {
                Text("Обновите или настройте API").foregroundStyle(.secondary)
            }
        }
        .navigationTitle("ML настройки")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: { Image(systemName: "arrow.clockwise") }
                .disabled(loading || !client.isConfigured)
            }
        }
        .task { if ml == nil && client.isConfigured { await load() } }
    }

    private func load() async {
        guard client.isConfigured else { errorMessage = "Настройте API"; return }
        loading = true
        errorMessage = nil
        defer { loading = false }
        do {
            ml = try await client.ml()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
