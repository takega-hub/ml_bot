import SwiftUI

struct PairsSettingsView: View {
    @ObservedObject var client: APIClient
    @State private var pairs: PairsResponse?
    @State private var errorMessage: String?
    @State private var loading = false
    @State private var addSymbol = ""
    @State private var message: String?

    var body: some View {
        Group {
            if loading && pairs == nil {
                ProgressView("Загрузка…")
            } else if let err = errorMessage {
                Text(err).foregroundStyle(.red).padding()
            } else if let p = pairs {
                List {
                    Section("Активные пары (макс \(p.maxActive))") {
                        ForEach(p.activeSymbols, id: \.self) { sym in
                            HStack {
                                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                                Text(sym)
                                Spacer()
                                if let cooldown = p.cooldowns[sym] {
                                    Text("❄️ \(cooldown.hoursLeft.map { String(format: "%.1f ч", $0) } ?? "")")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    Button("Снять") {
                                        Task { await removeCooldown(sym) }
                                    }
                                } else {
                                    Button("Выкл") {
                                        Task { await toggle(sym) }
                                    }
                                }
                            }
                        }
                        ForEach(p.knownSymbols.filter { !p.activeSymbols.contains($0) }, id: \.self) { sym in
                            HStack {
                                Image(systemName: "circle").foregroundStyle(.secondary)
                                Text(sym)
                                Spacer()
                                Button("Вкл") {
                                    Task { await toggle(sym) }
                                }
                                .disabled(p.activeSymbols.count >= p.maxActive)
                            }
                        }
                    }
                    Section("Добавить пару") {
                        HStack {
                            TextField("Символ (например XRPUSDT)", text: $addSymbol)
                                .textInputAutocapitalization(.characters)
                                .autocorrectionDisabled()
                            Button("Добавить") {
                                Task { await addPair() }
                            }
                            .disabled(addSymbol.count < 6)
                        }
                    }
                    if let msg = message {
                        Section {
                            Text(msg).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            } else {
                Text("Обновите или настройте API").foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Настройки пар")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: { Image(systemName: "arrow.clockwise") }
                .disabled(loading || !client.isConfigured)
            }
        }
        .task { if pairs == nil && client.isConfigured { await load() } }
    }

    private func load() async {
        guard client.isConfigured else { errorMessage = "Настройте API"; return }
        loading = true
        errorMessage = nil
        message = nil
        defer { loading = false }
        do {
            pairs = try await client.pairs()
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func toggle(_ symbol: String) async {
        do {
            let r = try await client.togglePair(symbol: symbol)
            message = r.enabled ? "\(r.symbol) включена" : "\(r.symbol) выключена"
            await load()
        } catch {
            message = error.localizedDescription
        }
    }

    private func removeCooldown(_ symbol: String) async {
        do {
            try await client.removeCooldown(symbol: symbol)
            message = "Разморозка снята для \(symbol)"
            await load()
        } catch {
            message = error.localizedDescription
        }
    }

    private func addPair() async {
        let sym = addSymbol.uppercased().trimmingCharacters(in: .whitespaces)
        guard sym.hasSuffix("USDT") else { message = "Символ должен заканчиваться на USDT"; return }
        do {
            let r = try await client.addPair(symbol: sym)
            message = r.message ?? (r.enabled == true ? "Пара \(sym) добавлена" : "Ok")
            addSymbol = ""
            await load()
        } catch {
            message = error.localizedDescription
        }
    }
}
