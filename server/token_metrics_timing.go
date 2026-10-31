package server

import (
	"time"

	"github.com/ollama/ollama/api"
)

func noteFirstGeneratedToken(first *time.Time, now time.Time, content, reasoning string, toolCalls []api.ToolCall) {
	if first == nil || !first.IsZero() {
		return
	}
	if content != "" || reasoning != "" || len(toolCalls) > 0 {
		*first = now
	}
}

// collectTiming derives the timing breakdown from the observed request
// timestamps and the runner metrics. requestStart is the wall-clock time the
// request was received; firstToken is the time of the first streamed content
// chunk (time.Time{} when the response did not stream any content chunk);
// metrics is the runner metrics reported with the Done response; parser, when
// non-nil, provides per-phase token counts for the duration split.
//
// TimeToFirstToken is firstToken − requestStart, clamped to 0.
// Parse falls back to the prompt eval duration reported by the runner — the
// time the runner spent evaluating the prompt (parsing it into the context
// window). Values that cannot be computed stay 0.
//
// The Answer/Reasoning/ToolCall split (summing to m.EvalDuration) is filled
// in by [splitDurations] from the parser state. When no parser state is
// available, all of m.EvalDuration falls to the answer phase.
func collectTiming(requestStart, firstToken time.Time, m api.Metrics, parser StateProvider) MetricsTiming {
	timing := MetricsTiming{}
	if !firstToken.IsZero() {
		if dtt := firstToken.Sub(requestStart); dtt > 0 {
			timing.TimeToFirstToken = dtt
		}
	}
	timing.Parse = m.PromptEvalDuration
	timing.Answer, timing.Reasoning, timing.ToolCall = splitDurations(m.EvalDuration, parser)
	return timing
}

// collectTimingWithGeneratedTokenCounts uses exact sampled-token phase labels
// for the duration split when they account for every runner-reported token.
func collectTimingWithGeneratedTokenCounts(
	requestStart, firstToken time.Time,
	m api.Metrics,
	parser StateProvider,
	counts *generatedTokenCounts,
) (MetricsTiming, bool) {
	timing := collectTiming(requestStart, firstToken, m, parser)
	if counts == nil || !counts.complete || counts.total != m.EvalCount {
		return timing, false
	}

	timing.Answer, timing.Reasoning, timing.ToolCall =
		splitDurationsByGeneratedTokenCounts(m.EvalDuration, counts)
	return timing, true
}

// splitDurationsByGeneratedTokenCounts estimates phase durations from exact
// sampled-token phase labels. Stream callback time can exceed EvalDuration.
func splitDurationsByGeneratedTokenCounts(total time.Duration, counts *generatedTokenCounts) (answer, reasoning, toolCall time.Duration) {
	answer = total
	if total <= 0 || counts == nil {
		return
	}

	totalCount := counts.answer + counts.reasoning + counts.toolCalls
	if totalCount <= 0 {
		return
	}

	reasoning = time.Duration(float64(total) * float64(counts.reasoning) / float64(totalCount))
	toolCall = time.Duration(float64(total) * float64(counts.toolCalls) / float64(totalCount))
	answer = total - reasoning - toolCall
	return
}
