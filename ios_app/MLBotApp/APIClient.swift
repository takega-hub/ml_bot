import Foundation

enum APIError: LocalizedError {
    case invalidURL
    case noCredentials
    case httpError(Int, String?)
    case decodingError(Error)

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Неверный URL сервера"
        case .noCredentials: return "Укажите URL и API ключ в настройках"
        case .httpError(let code, let msg): return "Ошибка \(code): \(msg ?? "нет ответа")"
        case .decodingError(let e): return "Ошибка данных: \(e.localizedDescription)"
        }
    }
}

@MainActor
final class APIClient: ObservableObject {
    private let baseURLKey = "mlbot_api_base_url"
    private let apiKeyKey = "mlbot_api_key"

    var baseURL: String {
        get { UserDefaults.standard.string(forKey: baseURLKey) ?? "" }
        set {
            UserDefaults.standard.set(newValue, forKey: baseURLKey)
            objectWillChange.send()
        }
    }

    var apiKey: String {
        get { UserDefaults.standard.string(forKey: apiKeyKey) ?? "" }
        set {
            UserDefaults.standard.set(newValue, forKey: apiKeyKey)
            objectWillChange.send()
        }
    }

    var isConfigured: Bool {
        !baseURL.isEmpty && !apiKey.isEmpty
    }

    private func makeRequest<T: Decodable>(
        path: String,
        method: String = "GET",
        body: Data? = nil
    ) async throws -> T {
        guard let url = URL(string: baseURL.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))) else {
            throw APIError.invalidURL
        }
        let fullURL = url.appendingPathComponent(path)
        var request = URLRequest(url: fullURL)
        request.httpMethod = method
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let body = body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = body
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw APIError.httpError(0, "Нет ответа")
        }
        if http.statusCode != 200 {
            let message = String(data: data, encoding: .utf8)
            throw APIError.httpError(http.statusCode, message)
        }
        do {
            let decoder = JSONDecoder()
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decodingError(error)
        }
    }

    func health() async throws -> [String: String] {
        guard let url = URL(string: baseURL.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))) else {
            throw APIError.invalidURL
        }
        let fullURL = url.appendingPathComponent("api/health")
        var request = URLRequest(url: fullURL)
        request.httpMethod = "GET"
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw APIError.httpError((response as? HTTPURLResponse)?.statusCode ?? 0, nil)
        }
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            return json
        }
        return ["status": "ok"]
    }

    func status() async throws -> StatusResponse {
        try await makeRequest(path: "api/status")
    }

    func dashboard() async throws -> DashboardResponse {
        try await makeRequest(path: "api/dashboard")
    }

    func settings() async throws -> SettingsResponse {
        try await makeRequest(path: "api/settings")
    }

    func start() async throws -> ControlResponse {
        try await makeRequest(path: "api/start", method: "POST")
    }

    func stop() async throws -> ControlResponse {
        try await makeRequest(path: "api/stop", method: "POST")
    }

    func pairs() async throws -> PairsResponse {
        try await makeRequest(path: "api/pairs")
    }

    func togglePair(symbol: String) async throws -> TogglePairResponse {
        let body = try! JSONSerialization.data(withJSONObject: ["symbol": symbol])
        return try await makeRequest(path: "api/pairs/toggle", method: "POST", body: body)
    }

    func removeCooldown(symbol: String) async throws {
        let body = try! JSONSerialization.data(withJSONObject: ["symbol": symbol])
        struct R: Decodable { let ok: Bool? }
        let _: R = try await makeRequest(path: "api/pairs/remove_cooldown", method: "POST", body: body)
    }

    func addPair(symbol: String) async throws -> AddPairResponse {
        let body = try! JSONSerialization.data(withJSONObject: ["symbol": symbol])
        return try await makeRequest(path: "api/pairs/add", method: "POST", body: body)
    }

    func risk() async throws -> RiskResponse {
        try await makeRequest(path: "api/risk")
    }

    func updateRisk(_ body: [String: Any]) async throws -> RiskResponse {
        let data = try JSONSerialization.data(withJSONObject: body)
        struct W: Decodable { let risk: RiskResponse }
        let w: W = try await makeRequest(path: "api/risk", method: "PUT", body: data)
        return w.risk
    }

    func ml() async throws -> MLResponse {
        try await makeRequest(path: "api/ml")
    }

    func updateML(_ body: [String: Any]) async throws -> MLResponse {
        let data = try JSONSerialization.data(withJSONObject: body)
        struct W: Decodable { let ml: MLResponse }
        let w: W = try await makeRequest(path: "api/ml", method: "PUT", body: data)
        return w.ml
    }

    func modelsList() async throws -> ModelsListResponse {
        try await makeRequest(path: "api/models")
    }

    func modelsForSymbol(_ symbol: String) async throws -> ModelsForSymbolResponse {
        try await makeRequest(path: "api/models/\(symbol)")
    }

    func applyModel(symbol: String, modelPath: String) async throws {
        struct Body: Encodable { let model_path: String }
        let body = try! JSONEncoder().encode(Body(model_path: modelPath))
        struct R: Decodable { let ok: Bool? }
        let _: R = try await makeRequest(path: "api/models/\(symbol)/apply", method: "POST", body: body)
    }

    func retrain(symbol: String) async throws {
        let body = try! JSONSerialization.data(withJSONObject: ["symbol": symbol])
        struct R: Decodable { let ok: Bool? }
        let _: R = try await makeRequest(path: "api/models/retrain", method: "POST", body: body)
    }

    func stats() async throws -> StatsResponse {
        try await makeRequest(path: "api/stats")
    }

    func historyTrades(limit: Int = 50) async throws -> HistoryTradesResponse {
        try await makeRequest(path: "api/history/trades?limit=\(limit)")
    }

    func historySignals(limit: Int = 30) async throws -> HistorySignalsResponse {
        try await makeRequest(path: "api/history/signals?limit=\(limit)")
    }

    func logs(type: String = "bot", lines: Int = 100) async throws -> LogsResponse {
        try await makeRequest(path: "api/logs?log_type=\(type)&lines=\(lines)")
    }

    func equityCurve() async throws -> EquityCurveResponse {
        try await makeRequest(path: "api/analytics/equity_curve")
    }

    func emergencyStopAll() async throws -> EmergencyStopResponse {
        try await makeRequest(path: "api/emergency/stop_all", method: "POST")
    }

    private func makeRequestRaw(path: String, method: String = "GET", body: Data? = nil) async throws -> Data {
        guard let url = URL(string: baseURL.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))) else {
            throw APIError.invalidURL
        }
        let fullURL = url.appendingPathComponent(path)
        var request = URLRequest(url: fullURL)
        request.httpMethod = method
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let body = body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = body
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw APIError.httpError((response as? HTTPURLResponse)?.statusCode ?? 0, String(data: data, encoding: .utf8))
        }
        return data
    }
}
