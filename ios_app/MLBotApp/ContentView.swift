import SwiftUI

struct ContentView: View {
    @StateObject private var client = APIClient()

    var body: some View {
        TabView {
            MainMenuView(client: client)
                .tabItem {
                    Label("Главная", systemImage: "square.grid.2x2")
                }
            DashboardView(client: client)
                .tabItem {
                    Label("Дашборд", systemImage: "chart.bar")
                }
            SettingsView(client: client)
                .tabItem {
                    Label("Настройки", systemImage: "gearshape")
                }
        }
    }
}

#Preview {
    ContentView()
}
