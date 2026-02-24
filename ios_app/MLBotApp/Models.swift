import Foundation

// MARK: - Status (GET /api/status)
struct StatusResponse: Codable {
    let isRunning: Bool
    let walletBalance: Double
    let availableBalance: Double
    let totalMargin: Double
    let positions: [Position]
    let strategies: [StrategyInfo]
    let activeSymbols: [String]
    let totalPnl: Double
    let winRate: Double
    let totalTrades: Int

    enum CodingKeys: String, CodingKey {
        case isRunning = "is_running"
        case walletBalance = "wallet_balance"
        case availableBalance = "available_balance"
        case totalMargin = "total_margin"
        case positions
        case strategies
        case activeSymbols = "active_symbols"
        case totalPnl = "total_pnl"
        case winRate = "win_rate"
        case totalTrades = "total_trades"
    }
}

struct Position: Codable {
    let symbol: String
    let side: String
    let size: Double
    let entry: Double
    let current: Double
    let pnl: Double
    let pnlPct: Double
    let leverage: Double
    let margin: Double
    let tp: Double?
    let sl: Double?

    enum CodingKeys: String, CodingKey {
        case symbol, side, size, entry, current, pnl, leverage, margin, tp, sl
        case pnlPct = "pnl_pct"
    }
}

struct StrategyInfo: Codable {
    let symbol: String
    let model: String?
    let mtf: Bool
    let model1h: String?
    let model15m: String?
    let cooldown: CooldownInfo?

    enum CodingKeys: String, CodingKey {
        case symbol, model, mtf, cooldown
        case model1h = "model_1h"
        case model15m = "model_15m"
    }
}

struct CooldownInfo: Codable {
    let hoursLeft: Double?
    let reason: String?

    enum CodingKeys: String, CodingKey {
        case reason
        case hoursLeft = "hours_left"
    }
}

// MARK: - Dashboard (GET /api/dashboard)
struct DashboardResponse: Codable {
    let walletBalance: Double
    let availableBalance: Double
    let totalMargin: Double
    let totalPnlPct: Double
    let openPositionsCount: Int
    let openPositionsPnl: Double
    let todayTrades: Int
    let todayPnl: Double
    let weekPnl: Double
    let weekWinrate: Double
    let weekTrades: Int
    let isRunning: Bool
    let activeSymbolsCount: Int

    enum CodingKeys: String, CodingKey {
        case walletBalance = "wallet_balance"
        case availableBalance = "available_balance"
        case totalMargin = "total_margin"
        case totalPnlPct = "total_pnl_pct"
        case openPositionsCount = "open_positions_count"
        case openPositionsPnl = "open_positions_pnl"
        case todayTrades = "today_trades"
        case todayPnl = "today_pnl"
        case weekPnl = "week_pnl"
        case weekWinrate = "week_winrate"
        case weekTrades = "week_trades"
        case isRunning = "is_running"
        case activeSymbolsCount = "active_symbols_count"
    }
}

// MARK: - Settings (GET /api/settings)
struct SettingsResponse: Codable {
    let activeSymbols: [String]
    let knownSymbols: [String]
    let leverage: Int
    let confidenceThreshold: Double
    let mtfConfidence1h: Double
    let mtfConfidence15m: Double

    enum CodingKeys: String, CodingKey {
        case activeSymbols = "active_symbols"
        case knownSymbols = "known_symbols"
        case leverage
        case confidenceThreshold = "confidence_threshold"
        case mtfConfidence1h = "mtf_confidence_1h"
        case mtfConfidence15m = "mtf_confidence_15m"
    }
}

// MARK: - Control (POST /api/start, /api/stop)
struct ControlResponse: Codable {
    let ok: Bool
    let isRunning: Bool

    enum CodingKeys: String, CodingKey {
        case ok
        case isRunning = "is_running"
    }
}

// MARK: - Pairs toggle/add responses
struct TogglePairResponse: Codable {
    let ok: Bool
    let symbol: String
    let enabled: Bool
}

struct AddPairResponse: Codable {
    let ok: Bool
    let symbol: String
    let enabled: Bool?
    let message: String?
}

// MARK: - Pairs (GET /api/pairs)
struct PairsResponse: Codable {
    let knownSymbols: [String]
    let activeSymbols: [String]
    let maxActive: Int
    let cooldowns: [String: CooldownInfo]

    enum CodingKeys: String, CodingKey {
        case knownSymbols = "known_symbols"
        case activeSymbols = "active_symbols"
        case maxActive = "max_active"
        case cooldowns
    }
}

// MARK: - Risk (GET/PUT /api/risk) — flat dict
struct RiskResponse: Codable {
    let marginPctBalance: Double?
    let baseOrderUsd: Double?
    let stopLossPct: Double?
    let takeProfitPct: Double?
    let enableTrailingStop: Bool?
    let trailingStopActivationPct: Double?
    let trailingStopDistancePct: Double?
    let enablePartialClose: Bool?
    let enableBreakeven: Bool?
    let breakevenLevel1ActivationPct: Double?
    let breakevenLevel1SlPct: Double?
    let breakevenLevel2ActivationPct: Double?
    let breakevenLevel2SlPct: Double?
    let enableLossCooldown: Bool?
    let feeRate: Double?
    let midTermTpPct: Double?
    let longTermTpPct: Double?
    let longTermSlPct: Double?
    let longTermIgnoreReverse: Bool?
    let dcaEnabled: Bool?
    let dcaDrawdownPct: Double?
    let dcaMaxAdds: Int?
    let dcaMinConfidence: Double?
    let reverseOnStrongSignal: Bool?
    let reverseMinConfidence: Double?

    enum CodingKeys: String, CodingKey {
        case marginPctBalance = "margin_pct_balance"
        case baseOrderUsd = "base_order_usd"
        case stopLossPct = "stop_loss_pct"
        case takeProfitPct = "take_profit_pct"
        case enableTrailingStop = "enable_trailing_stop"
        case trailingStopActivationPct = "trailing_stop_activation_pct"
        case trailingStopDistancePct = "trailing_stop_distance_pct"
        case enablePartialClose = "enable_partial_close"
        case enableBreakeven = "enable_breakeven"
        case breakevenLevel1ActivationPct = "breakeven_level1_activation_pct"
        case breakevenLevel1SlPct = "breakeven_level1_sl_pct"
        case breakevenLevel2ActivationPct = "breakeven_level2_activation_pct"
        case breakevenLevel2SlPct = "breakeven_level2_sl_pct"
        case enableLossCooldown = "enable_loss_cooldown"
        case feeRate = "fee_rate"
        case midTermTpPct = "mid_term_tp_pct"
        case longTermTpPct = "long_term_tp_pct"
        case longTermSlPct = "long_term_sl_pct"
        case longTermIgnoreReverse = "long_term_ignore_reverse"
        case dcaEnabled = "dca_enabled"
        case dcaDrawdownPct = "dca_drawdown_pct"
        case dcaMaxAdds = "dca_max_adds"
        case dcaMinConfidence = "dca_min_confidence"
        case reverseOnStrongSignal = "reverse_on_strong_signal"
        case reverseMinConfidence = "reverse_min_confidence"
    }
}

// MARK: - ML (GET/PUT /api/ml)
struct MLResponse: Codable {
    let useMtfStrategy: Bool?
    let mtfConfidenceThreshold1h: Double?
    let mtfConfidenceThreshold15m: Double?
    let mtfAlignmentMode: String?
    let atrFilterEnabled: Bool?
    let autoOptimizeStrategies: Bool?
    let autoOptimizeDay: String?
    let autoOptimizeHour: Int?
    let useFixedSlFromRisk: Bool?
    let confidenceThreshold: Double?
    let minConfidenceForTrade: Double?

    enum CodingKeys: String, CodingKey {
        case useMtfStrategy = "use_mtf_strategy"
        case mtfConfidenceThreshold1h = "mtf_confidence_threshold_1h"
        case mtfConfidenceThreshold15m = "mtf_confidence_threshold_15m"
        case mtfAlignmentMode = "mtf_alignment_mode"
        case atrFilterEnabled = "atr_filter_enabled"
        case autoOptimizeStrategies = "auto_optimize_strategies"
        case autoOptimizeDay = "auto_optimize_day"
        case autoOptimizeHour = "auto_optimize_hour"
        case useFixedSlFromRisk = "use_fixed_sl_from_risk"
        case confidenceThreshold = "confidence_threshold"
        case minConfidenceForTrade = "min_confidence_for_trade"
    }
}

// MARK: - Models (GET /api/models, GET /api/models/{symbol})
struct ModelsListResponse: Codable {
    let symbols: [SymbolModelInfo]
}

struct SymbolModelInfo: Codable {
    let symbol: String
    let modelPath: String?
    let modelName: String?

    enum CodingKeys: String, CodingKey {
        case symbol
        case modelPath = "model_path"
        case modelName = "model_name"
    }
}

struct ModelsForSymbolResponse: Codable {
    let symbol: String
    let models: [ModelWithTest]
    let current: String?
}

struct ModelWithTest: Codable {
    let index: Int
    let path: String
    let name: String
    let current: Bool
    let test: [String: Double]?
}

// MARK: - History (GET /api/history/trades, /api/history/signals)
struct HistoryTradesResponse: Codable {
    let trades: [TradeRecordDTO]
}

struct TradeRecordDTO: Codable {
    let symbol: String
    let side: String
    let entryPrice: Double
    let exitPrice: Double?
    let qty: Double
    let pnlUsd: Double
    let pnlPct: Double
    let entryTime: String?
    let exitTime: String?
    let status: String?

    enum CodingKeys: String, CodingKey {
        case symbol, side, qty, status
        case entryPrice = "entry_price"
        case exitPrice = "exit_price"
        case pnlUsd = "pnl_usd"
        case pnlPct = "pnl_pct"
        case entryTime = "entry_time"
        case exitTime = "exit_time"
    }
}

struct HistorySignalsResponse: Codable {
    let signals: [SignalRecordDTO]
}

struct SignalRecordDTO: Codable {
    let timestamp: String
    let symbol: String
    let action: String
    let price: Double
    let confidence: Double
    let reason: String
}

// MARK: - Logs (GET /api/logs)
struct LogsResponse: Codable {
    let lines: [String]
    let path: String?
}

// MARK: - Stats (GET /api/stats)
struct StatsResponse: Codable {
    let totalPnl: Double
    let winRate: Double
    let totalTrades: Int
    let closedCount: Int
    let openCount: Int
    let winsCount: Int
    let lossesCount: Int
    let avgWin: Double?
    let avgLoss: Double?

    enum CodingKeys: String, CodingKey {
        case totalPnl = "total_pnl"
        case winRate = "win_rate"
        case totalTrades = "total_trades"
        case closedCount = "closed_count"
        case openCount = "open_count"
        case winsCount = "wins_count"
        case lossesCount = "losses_count"
        case avgWin = "avg_win"
        case avgLoss = "avg_loss"
    }
}

// MARK: - Analytics (GET /api/analytics/equity_curve)
struct EquityCurveResponse: Codable {
    let points: [EquityPoint]
    let totalPnl: Double

    enum CodingKeys: String, CodingKey {
        case points
        case totalPnl = "total_pnl"
    }
}

struct EquityPoint: Codable {
    let time: String
    let pnlCum: Double
    let tradePnl: Double

    enum CodingKeys: String, CodingKey {
        case time
        case pnlCum = "pnl_cum"
        case tradePnl = "trade_pnl"
    }
}

// MARK: - Emergency (POST /api/emergency/stop_all)
struct EmergencyStopResponse: Codable {
    let ok: Bool
    let closedPositions: [String]
    let message: String?

    enum CodingKeys: String, CodingKey {
        case ok
        case closedPositions = "closed_positions"
        case message
    }
}
