import SwiftUI

/// Главное меню как в Telegram-админке: сетка кнопок.
struct MainMenuView: View {
    @ObservedObject var client: APIClient
    @State private var isRunning: Bool?
    @State private var errorMessage: String?

    private let columns = [GridItem(.flexible()), GridItem(.flexible())]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if let err = errorMessage {
                        Text(err)
                            .font(.caption)
                            .foregroundStyle(.red)
                            .padding(.horizontal)
                    }
                    LazyVGrid(columns: columns, spacing: 12) {
                        Button {
                            Task { await doStart() }
                        } label: {
                            MenuButtonLabel(title: "Старт", icon: "play.circle.fill", color: .green)
                        }
                        .buttonStyle(.plain)
                        .disabled(!client.isConfigured)

                        Button {
                            Task { await doStop() }
                        } label: {
                            MenuButtonLabel(title: "Стоп", icon: "stop.circle.fill", color: .red)
                        }
                        .buttonStyle(.plain)
                        .disabled(!client.isConfigured)

                        NavigationLink(destination: StatusView(client: client)) {
                            MenuButtonLabel(title: "Статус", icon: "chart.bar.doc.plain", color: .blue)
                        }
                        .buttonStyle(.plain)

                        NavigationLink(destination: StatsView(client: client)) {
                            MenuButtonLabel(title: "Статистика", icon: "chart.line.uptrend.xyaxis", color: .orange)
                        }
                        .buttonStyle(.plain)

                        NavigationLink(destination: PairsSettingsView(client: client)) {
                            MenuButtonLabel(title: "Настройки пар", icon: "gearshape.2", color: .gray)
                        }
                        .buttonStyle(.plain)

                        NavigationLink(destination: ModelsListView(client: client)) {
                            MenuButtonLabel(title: "Модели", icon: "cpu", color: .purple)
                        }
                        .buttonStyle(.plain)

                        NavigationLink(destination: RiskSettingsView(client: client)) {
                            MenuButtonLabel(title: "Настройки риска", icon: "shield.lefthalf.filled", color: .indigo)
                        }
                        .buttonStyle(.plain)

                        NavigationLink(destination: MLSettingsView(client: client)) {
                            MenuButtonLabel(title: "ML настройки", icon: "brain.head.profile", color: .teal)
                        }
                        .buttonStyle(.plain)

                        NavigationLink(destination: HistoryView(client: client)) {
                            MenuButtonLabel(title: "История", icon: "doc.text.magnifyingglass", color: .brown)
                        }
                        .buttonStyle(.plain)

                        NavigationLink(destination: EmergencyView(client: client)) {
                            MenuButtonLabel(title: "Экстренные", icon: "exclamationmark.triangle.fill", color: .red)
                        }
                        .buttonStyle(.plain)

                        NavigationLink(destination: AnalyticsView(client: client)) {
                            MenuButtonLabel(title: "Аналитика", icon: "chart.xyaxis.line", color: .mint)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding()
                }
            }
            .navigationTitle("ML Trading Bot")
            .navigationBarTitleDisplayMode(.inline)
            .task {
                if client.isConfigured && isRunning == nil {
                    await refreshState()
                }
            }
        }
    }

    private func doStart() async {
        guard client.isConfigured else { errorMessage = "Настройте API"; return }
        errorMessage = nil
        do {
            _ = try await client.start()
            isRunning = true
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func doStop() async {
        guard client.isConfigured else { return }
        errorMessage = nil
        do {
            _ = try await client.stop()
            isRunning = false
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func refreshState() async {
        guard client.isConfigured else { return }
        do {
            let s = try await client.status()
            isRunning = s.isRunning
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

struct MenuButtonLabel: View {
    let title: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundStyle(color)
            Text(title)
                .font(.caption)
                .multilineTextAlignment(.center)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .background(color.opacity(0.15))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
