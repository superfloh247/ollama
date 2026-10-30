package server

import (
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestCollectTiming(t *testing.T) {
	requestStart := time.Date(2026, 10, 14, 12, 0, 0, 0, time.UTC)
	firstToken := requestStart.Add(250 * time.Millisecond)

	m := api.Metrics{
		PromptEvalCount:    100,
		PromptEvalDuration: 300 * time.Millisecond,
		EvalCount:          30,
		EvalDuration:       time.Second,
	}

	t.Run("TimeToFirstToken is first token minus request start", func(t *testing.T) {
		timing := collectTiming(requestStart, firstToken, m, nil)
		if timing.TimeToFirstToken != 250*time.Millisecond {
			t.Errorf("TimeToFirstToken = %v, want %v", timing.TimeToFirstToken, 250*time.Millisecond)
		}
	})

	t.Run("Parse falls back to the prompt eval duration", func(t *testing.T) {
		timing := collectTiming(requestStart, firstToken, m, nil)
		if timing.Parse != 300*time.Millisecond {
			t.Errorf("Parse = %v, want %v", timing.Parse, 300*time.Millisecond)
		}
	})

	t.Run("no streamed token yields zero TimeToFirstToken", func(t *testing.T) {
		timing := collectTiming(requestStart, time.Time{}, m, nil)
		if timing.TimeToFirstToken != 0 {
			t.Errorf("TimeToFirstToken = %v, want 0", timing.TimeToFirstToken)
		}
	})

	t.Run("clock jump (token before request start) clamps to zero", func(t *testing.T) {
		timing := collectTiming(requestStart, requestStart.Add(-50*time.Millisecond), m, nil)
		if timing.TimeToFirstToken != 0 {
			t.Errorf("TimeToFirstToken = %v, want 0", timing.TimeToFirstToken)
		}
	})

	t.Run("no metrics yield zero timings", func(t *testing.T) {
		timing := collectTiming(requestStart, time.Time{}, api.Metrics{}, nil)
		if timing.TimeToFirstToken != 0 || timing.Parse != 0 {
			t.Errorf("timing = %+v, want all zero", timing)
		}
	})
}
