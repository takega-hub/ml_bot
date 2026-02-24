import SwiftUI

struct ModelsListView: View {
    @ObservedObject var client: APIClient
    @State private var list: ModelsListResponse?
    @State private var errorMessage: String?
    @State private var loading = false

    var body: some View {
        Group {
            if loading && list == nil {
                ProgressView("Загрузка…")
            } else if let err = errorMessage {
                Text(err).foregroundStyle(.red).padding()
            } else if let list = list, !list.symbols.isEmpty {
                List {
                    ForEach(list.symbols, id: \.symbol) { item in
                        NavigationLink(destination: ModelSelectionView(client: client, symbol: item.symbol)) {
                            HStack {
                                Text(item.symbol)
                                Spacer()
                                if let name = item.modelName {
                                    Text(name)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                } else {
                                    Text("Авто").font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            } else {
                Text("Нет активных пар или обновите API").foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Модели")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: { Image(systemName: "arrow.clockwise") }
                .disabled(loading || !client.isConfigured)
            }
        }
        .task { if list == nil && client.isConfigured { await load() } }
    }

    private func load() async {
        guard client.isConfigured else { errorMessage = "Настройте API"; return }
        loading = true
        errorMessage = nil
        defer { loading = false }
        do {
            list = try await client.modelsList()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

struct ModelSelectionView: View {
    @ObservedObject var client: APIClient
    let symbol: String
    @State private var data: ModelsForSymbolResponse?
    @State private var errorMessage: String?
    @State private var loading = false
    @State private var message: String?

    var body: some View {
        Group {
            if loading && data == nil {
                ProgressView("Загрузка…")
            } else if let err = errorMessage {
                Text(err).foregroundStyle(.red).padding()
            } else if let d = data {
                List {
                    ForEach(d.models, id: \.path) { m in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text(m.name)
                                    if m.current {
                                        Text("✓").foregroundStyle(.green)
                                    }
                                }
                                if let test = m.test, let pnl = test["total_pnl_pct"], let wr = test["win_rate"] {
                                    Text("PnL: \(pnl >= 0 ? "+" : "")\(pnl, specifier: "%.2f")% | WR: \(wr, specifier: "%.1f")%")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            Spacer()
                            if !m.current {
                                Button("Применить") {
                                    Task { await apply(m.path) }
                                }
                            }
                        }
                    }
                    Section {
                        Button("Переобучить модель") {
                            Task { await retrain() }
                        }
                        .disabled(loading)
                    }
                    if let msg = message {
                        Section {
                            Text(msg).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            } else {
                Text("Обновите").foregroundStyle(.secondary)
            }
        }
        .navigationTitle(symbol)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: { Image(systemName: "arrow.clockwise") }
                .disabled(loading || !client.isConfigured)
            }
        }
        .task { if data == nil { await load() } }
    }

    private func load() async {
        guard client.isConfigured else { errorMessage = "Настройте API"; return }
        loading = true
        errorMessage = nil
        message = nil
        defer { loading = false }
        do {
            data = try await client.modelsForSymbol(symbol)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func apply(_ path: String) async {
        do {
            try await client.applyModel(symbol: symbol, modelPath: path)
            message = "Модель применена"
            await load()
        } catch {
            message = error.localizedDescription
        }
    }

    private func retrain() async {
        do {
            try await client.retrain(symbol: symbol)
            message = "Переобучение запущено в фоне"
        } catch {
            message = error.localizedDescription
        }
    }
}
