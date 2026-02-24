import SwiftUI

struct HistoryView: View {
    @ObservedObject var client: APIClient
    @State private var segment = 0
    @State private var trades: [TradeRecordDTO] = []
    @State private var signals: [SignalRecordDTO] = []
    @State private var logLines: [String] = []
    @State private var logType = "bot"
    @State private var errorMessage: String?
    @State private var loading = false

    var body: some View {
        VStack(spacing: 0) {
            Picker("", selection: $segment) {
                Text("Сделки").tag(0)
                Text("Сигналы").tag(1)
                Text("Логи").tag(2)
            }
            .pickerStyle(.segmented)
            .padding()

            Group {
                if loading {
                    ProgressView("Загрузка…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let err = errorMessage {
                    Text(err).foregroundStyle(.red).padding()
                } else if segment == 0 {
                    tradesList
                } else if segment == 1 {
                    signalsList
                } else {
                    logsList
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .navigationTitle("История")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: { Image(systemName: "arrow.clockwise") }
                .disabled(loading || !client.isConfigured)
            }
        }
        .onChange(of: segment) { _ in Task { await load() } }
        .task { if client.isConfigured { await load() } }
    }

    private var tradesList: some View {
        List {
            ForEach(Array(trades.enumerated()), id: \.offset) { _, t in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(t.symbol).fontWeight(.semibold)
                        Text(t.side).foregroundStyle(t.side == "Buy" ? .green : .red)
                        Spacer()
                        Text("\(t.pnlUsd >= 0 ? "+" : "")$\(t.pnlUsd, specifier: "%.2f")")
                            .foregroundStyle(t.pnlUsd >= 0 ? .green : .red)
                    }
                    if let et = t.exitTime, let idx = et.firstIndex(of: "T") {
                        Text(String(et[..<idx]) + " " + String(et[et.index(idx, offsetBy: 1)...]).prefix(8))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.vertical, 4)
            }
        }
    }

    private var signalsList: some View {
        List {
            ForEach(Array(signals.enumerated()), id: \.offset) { _, s in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(s.symbol) \(s.action)")
                        Text(s.reason).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text("\(Int(s.confidence * 100))%")
                        .font(.caption)
                }
                .padding(.vertical, 4)
            }
        }
    }

    private var logsList: some View {
        VStack(alignment: .leading, spacing: 0) {
            Picker("Лог", selection: $logType) {
                Text("Бот").tag("bot")
                Text("Сделки").tag("trades")
                Text("Сигналы").tag("signals")
                Text("Ошибки").tag("errors")
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .onChange(of: logType) { _ in Task { await loadLogs() } }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(logLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(.caption, design: .monospaced))
                    }
                }
                .padding()
            }
        }
    }

    private func load() async {
        guard client.isConfigured else { errorMessage = "Настройте API"; return }
        loading = true
        errorMessage = nil
        defer { loading = false }
        if segment == 0 {
            do {
                let r = try await client.historyTrades()
                trades = r.trades
            } catch {
                errorMessage = error.localizedDescription
            }
        } else if segment == 1 {
            do {
                let r = try await client.historySignals()
                signals = r.signals
            } catch {
                errorMessage = error.localizedDescription
            }
        } else {
            await loadLogs()
        }
    }

    private func loadLogs() async {
        do {
            let r = try await client.logs(type: logType, lines: 80)
            logLines = r.lines
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
