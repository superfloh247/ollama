package server

import (
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestNoteFirstGeneratedToken(t *testing.T) {
	first := time.Time{}
	now := time.Date(2026, 10, 30, 12, 0, 0, 0, time.UTC)

	t.Run("reasoning-only output starts TTFT", func(t *testing.T) {
		noteFirstGeneratedToken(&first, now, "", "reasoning", nil)
		if !first.Equal(now) {
			t.Errorf("first token time = %v, want %v", first, now)
		}
	})

	t.Run("later content does not overwrite the first generated token", func(t *testing.T) {
		noteFirstGeneratedToken(&first, now.Add(time.Second), "answer", "", nil)
		if !first.Equal(now) {
			t.Errorf("first token time = %v, want unchanged %v", first, now)
		}
	})

	t.Run("tool-only output starts TTFT", func(t *testing.T) {
		first = time.Time{}
		noteFirstGeneratedToken(&first, now, "", "", []api.ToolCall{{}})
		if !first.Equal(now) {
			t.Errorf("first token time = %v, want %v", first, now)
		}
	})

	t.Run("empty chunks and nil target do not start TTFT", func(t *testing.T) {
		first = time.Time{}
		noteFirstGeneratedToken(&first, now, "", "", nil)
		noteFirstGeneratedToken(nil, now, "answer", "", nil)
		if !first.IsZero() {
			t.Errorf("first token time = %v, want zero", first)
		}
	})
}

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

func TestCollectTimingUsesExactGeneratedTokenPhases(t *testing.T) {
	requestStart := time.Date(2026, 10, 14, 12, 0, 0, 0, time.UTC)
	metrics := api.Metrics{EvalCount: 10, EvalDuration: time.Second}
	parser := fakeSplitParser{thinkingTokens: 10, toolCallTokens: 0, answerTokens: 10}
	counts := &generatedTokenCounts{
		reasoning: 2,
		toolCalls: 3,
		answer:    5,
		total:     10,
		complete:  true,
	}

	timing, exact := collectTimingWithGeneratedTokenCounts(requestStart, time.Time{}, metrics, parser, counts)
	if !exact {
		t.Fatal("expected exact generated token phases to be used")
	}
	if timing.Reasoning != 200*time.Millisecond || timing.ToolCall != 300*time.Millisecond || timing.Answer != 500*time.Millisecond {
		t.Errorf("timing = %+v, want answer=500ms reasoning=200ms toolCall=300ms", timing)
	}
	if timing.Reasoning+timing.ToolCall+timing.Answer != metrics.EvalDuration {
		t.Fatalf("phase durations do not sum to eval duration: %+v", timing)
	}
}
