# AI Research Runbook

## 1) Быстрый маршрут чтения эксперимента

1. Открыть `/api/ai/research/status` и выбрать нужный `experiment_id`.
2. Проверить:
   - `status`, `campaign.status`, `campaign.next_experiment_id`
   - `analysis_summary`
   - `recommendation`, `champion_challenger`
3. Для устойчивости посмотреть:
   - `oos_metrics`
   - `drift_signals`
   - `stress_results`
4. Для причин решения открыть:
   - `results.decision_trace`
   - `research_notebook.summary`
   - `research_notebook.entries[*]`

## 2) Интерпретация ключевых блоков

### oos_metrics

- `passed=true` — OOS-фильтр пройден для рекомендованной тактики.
- `evaluation.score` — вспомогательная оценка на OOS.
- Если `passed=false`, модель не рекомендуется к замене champion, даже при хорошем in-sample.

### drift_signals

- `level`:
  - `low` — признаков деградации мало;
  - `medium` — есть признаки смещения, нужен контроль;
  - `high` — высокая вероятность деградации/нестабильности.
- `drift_score` — агрегированный индикатор риска деградации.
- `signals.wf_instability=true` — walk-forward нестабилен.
- `signals.oos_gate_failed=true` — OOS-гейт не пройден.

### stress_results

- `stress_passed=true` — большинство стресс-сценариев пройдено.
- `robustness_score` — доля пройденных стресс-сценариев.
- `scenarios[*]` — детализация по сценариям:
  - `stressed_pnl_pct`
  - `stressed_max_drawdown_pct`
  - `stressed_profit_factor`
  - `passed`

### decision_trace

- `chosen_tactic` и `chosen_score` — итог выбора.
- `why_this_tactic` — объяснение текущего шага.
- `why_not_others` — почему альтернативы не выбраны.
- `candidate_rankings` — ранжирование кандидатов по итоговому скорингу.

### research_notebook (v2)

- `summary.stop_reason` — почему цепочка остановилась/перешла к следующему шагу.
- `entries[*].delta_vs_prev` — дельта против предыдущей итерации.
- `entries[*].expected_vs_actual` — структурная сверка ожидания и факта.
- `entries[*].stop_reason` — локальная причина остановки на итерации.

## 3) Операционные правила принятия решения

- `recommendation=replace` допустимо, если одновременно:
  - quality gates пройдены;
  - OOS-гейты не провалены;
  - walk-forward стабилен;
  - stress_results не показывает критической деградации.
- При `drift_signals.level=high`:
  - не переводить в production без дополнительной валидации;
  - запускать дополнительный цикл с ужесточёнными настройками.
- При `stress_passed=false`:
  - рассматривать тактику как нестабильную в реальных условиях исполнения.

## 4) Частые stop_reason

- `scheduled_next` — автоматически назначена следующая итерация.
- `max_steps_reached` — достигнут лимит шага кампании.
- `manual_stop` — остановлено вручную.
- `failed_to_schedule_next` — переход к следующей итерации не выполнен.
- `failed_to_schedule_next_exception` — ошибка в оркестрации следующего шага.
- `no_next_iteration` — цепочка завершилась без планирования следующего шага.

## 5) Минимальный чеклист перед apply

- Проверить `champion_challenger.challenger.gates_passed == true`.
- Проверить `oos_metrics.passed == true`.
- Проверить `drift_signals.level != high`.
- Проверить `stress_results.stress_passed == true`.
- Проверить в notebook, что `delta_vs_prev` не указывает на деградацию.
