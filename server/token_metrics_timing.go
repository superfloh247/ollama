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
