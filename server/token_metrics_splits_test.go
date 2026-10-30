package server

import (
	"testing"
	"time"
)

// fakeSplitParser is a [StateProvider] with simulated parser state: fixed
// reasoning / tool-call token counts as reported by a parser at the end of a
// request.
type fakeSplitParser struct {
	thinkingTokens int
	toolCallTokens int
}

func (p fakeSplitParser) ThinkingTokens() int { return p.thinkingTokens }
func (p fakeSplitParser) ToolCallTokens() int { return p.toolCallTokens }

func TestSplitGeneratedTokens(t *testing.T) {
	t.Run("no parser attributes everything to the answer", func(t *testing.T) {
		reasoning, toolCalls, answer := splitGeneratedTokens(50, nil)
		if reasoning != 0 || toolCalls != 0 || answer != 50 {
			t.Errorf("splitGeneratedTokens(50, nil) = (%d, %d, %d), want (0, 0, 50)", reasoning, toolCalls, answer)
		}
	})

	t.Run("reasoning and tool-call tokens are split out of the total", func(t *testing.T) {
		parser := fakeSplitParser{thinkingTokens: 10, toolCallTokens: 5}
		reasoning, toolCalls, answer := splitGeneratedTokens(30, parser)
		if reasoning != 10 || toolCalls != 5 || answer != 15 {
			t.Errorf("split = (%d, %d, %d), want (10, 5, 15)", reasoning, toolCalls, answer)
		}
		if sum := reasoning + toolCalls + answer; sum != 30 {
			t.Errorf("sum = %d, want 30", sum)
		}
	})

	t.Run("no reasoning and no tool calls", func(t *testing.T) {
		parser := fakeSplitParser{}
		reasoning, toolCalls, answer := splitGeneratedTokens(20, parser)
		if reasoning != 0 || toolCalls != 0 || answer != 20 {
			t.Errorf("split = (%d, %d, %d), want (0, 0, 20)", reasoning, toolCalls, answer)
		}
	})

	t.Run("negative parser counts are clamped to zero", func(t *testing.T) {
		parser := fakeSplitParser{thinkingTokens: -3, toolCallTokens: -2}
		reasoning, toolCalls, answer := splitGeneratedTokens(12, parser)
		if reasoning != 0 || toolCalls != 0 || answer != 12 {
			t.Errorf("split = (%d, %d, %d), want (0, 0, 12)", reasoning, toolCalls, answer)
		}
	})

	t.Run("parser counts exceeding the total are clamped to keep the sum", func(t *testing.T) {
		// The parser may have seen more content than the runner has reported as
		// EvalCount so far; the attributed counts must then be capped so that
		// the three parts still sum to evalCount.
		parser := fakeSplitParser{thinkingTokens: 20, toolCallTokens: 10}
		reasoning, toolCalls, answer := splitGeneratedTokens(15, parser)
		if reasoning != 15 || toolCalls != 0 || answer != 0 {
			t.Errorf("split = (%d, %d, %d), want (15, 0, 0)", reasoning, toolCalls, answer)
		}
		if sum := reasoning + toolCalls + answer; sum != 15 {
			t.Errorf("sum = %d, want 15", sum)
		}
	})

	t.Run("zero tokens with no parser", func(t *testing.T) {
		reasoning, toolCalls, answer := splitGeneratedTokens(0, nil)
		if reasoning != 0 || toolCalls != 0 || answer != 0 {
			t.Errorf("splitGeneratedTokens(0, nil) = (%d, %d, %d), want (0, 0, 0)", reasoning, toolCalls, answer)
		}
	})
}

func TestSplitDurations(t *testing.T) {
	oneSecond := time.Second

	t.Run("no parser attributes everything to the answer", func(t *testing.T) {
		answer, reasoning, toolCall := splitDurations(oneSecond, nil)
		if answer != oneSecond || reasoning != 0 || toolCall != 0 {
			t.Errorf("splitDurations(1s, nil) = (%v, %v, %v), want (1s, 0, 0)", answer, reasoning, toolCall)
		}
	})

	t.Run("duration split proportional to token distribution", func(t *testing.T) {
		// 60/40 split: thinking=3, toolCalls=2, totalAttributed=5
		parser := fakeSplitParser{thinkingTokens: 3, toolCallTokens: 2}
		totalEval := 500 * time.Millisecond
		answer, reasoning, toolCall := splitDurations(totalEval, parser)
		// reasoning = 500ms * 3/5 = 300ms, toolCall = 500ms * 2/5 = 200ms
		// answer = 500ms - 300ms - 200ms = 0
		if reasoning != 300*time.Millisecond {
			t.Errorf("reasoning = %v, want 300ms", reasoning)
		}
		if toolCall != 200*time.Millisecond {
			t.Errorf("toolCall = %v, want 200ms", toolCall)
		}
		if answer != 0 {
			t.Errorf("answer = %v, want 0", answer)
		}
		if sum := reasoning + toolCall + answer; sum != totalEval {
			t.Errorf("sum = %v, want %v", sum, totalEval)
		}
	})

	t.Run("no reasoning and no tool calls", func(t *testing.T) {
		parser := fakeSplitParser{}
		answer, reasoning, toolCall := splitDurations(oneSecond, parser)
		if answer != oneSecond || reasoning != 0 || toolCall != 0 {
			t.Errorf("split = (%v, %v, %v), want (1s, 0, 0)", answer, reasoning, toolCall)
		}
	})

	t.Run("negative parser counts are clamped to zero", func(t *testing.T) {
		parser := fakeSplitParser{thinkingTokens: -3, toolCallTokens: -2}
		answer, reasoning, toolCall := splitDurations(oneSecond, parser)
		if answer != oneSecond || reasoning != 0 || toolCall != 0 {
			t.Errorf("split = (%v, %v, %v), want (1s, 0, 0)", answer, reasoning, toolCall)
		}
	})

	t.Run("zero eval duration", func(t *testing.T) {
		parser := fakeSplitParser{thinkingTokens: 5, toolCallTokens: 5}
		answer, reasoning, toolCall := splitDurations(0, parser)
		if answer != 0 || reasoning != 0 || toolCall != 0 {
			t.Errorf("splitDurations(0, parser) = (%v, %v, %v), want (0, 0, 0)", answer, reasoning, toolCall)
		}
	})

	t.Run("only reasoning tokens", func(t *testing.T) {
		parser := fakeSplitParser{thinkingTokens: 10, toolCallTokens: 0}
		answer, reasoning, toolCall := splitDurations(500*time.Millisecond, parser)
		if reasoning != 500*time.Millisecond || toolCall != 0 || answer != 0 {
			t.Errorf("split = (%v, %v, %v), want (0, 500ms, 0)", answer, reasoning, toolCall)
		}
	})

	t.Run("only tool call tokens", func(t *testing.T) {
		parser := fakeSplitParser{thinkingTokens: 0, toolCallTokens: 10}
		answer, reasoning, toolCall := splitDurations(600*time.Millisecond, parser)
		if reasoning != 0 || toolCall != 600*time.Millisecond || answer != 0 {
			t.Errorf("split = (%v, %v, %v), want (0, 0, 600ms)", answer, reasoning, toolCall)
		}
	})

	t.Run("uneven distribution with rounding clamping", func(t *testing.T) {
		// 70/30 split: thinking=7, toolCalls=3, total=10
		// 100ms * 7/10 = 70ms, 100ms * 3/10 = 30ms, answer = 100 - 70 - 30 = 0
		parser := fakeSplitParser{thinkingTokens: 7, toolCallTokens: 3}
		totalEval := 100 * time.Millisecond
		answer, reasoning, toolCall := splitDurations(totalEval, parser)
		if reasoning != 70*time.Millisecond {
			t.Errorf("reasoning = %v, want 70ms", reasoning)
		}
		if toolCall != 30*time.Millisecond {
			t.Errorf("toolCall = %v, want 30ms", toolCall)
		}
		if answer != 0 {
			t.Errorf("answer = %v, want 0", answer)
		}
		if sum := reasoning + toolCall + answer; sum != totalEval {
			t.Errorf("sum = %v, want %v", sum, totalEval)
		}
	})
}
