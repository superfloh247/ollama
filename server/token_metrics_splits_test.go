package server

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// fakeSplitParser is a [StateProvider] with fixed per-phase content byte counts.
type fakeSplitParser struct {
	thinkingTokens int
	toolCallTokens int
	answerTokens   int
}

func (p fakeSplitParser) ThinkingTokens() int { return p.thinkingTokens }
func (p fakeSplitParser) ToolCallTokens() int { return p.toolCallTokens }
func (p fakeSplitParser) AnswerTokens() int   { return p.answerTokens }

func TestCountGeneratedTokenPhasesUsesTokenizer(t *testing.T) {
	text := generatedTokenText{}
	text.addReasoning("reasoning bytes")
	text.toolCalls.WriteString("tool call bytes")
	text.addAnswer("answer bytes")

	counts, err := countGeneratedTokenPhases(context.Background(), 50,
		func(_ context.Context, content string) ([]int, error) {
			switch content {
			case "reasoning bytes":
				return []int{1, 2, 3}, nil
			case "tool call bytes":
				return []int{1}, nil
			case "answer bytes":
				return []int{1, 2, 3, 4, 5, 6}, nil
			default:
				t.Fatalf("unexpected tokenizer input %q", content)
				return nil, nil
			}
		}, text)
	if err != nil {
		t.Fatal(err)
	}
	if counts != (generatedTokenCounts{reasoning: 15, toolCalls: 5, answer: 30}) {
		t.Errorf("counts = %+v, want reasoning 15, tool calls 5, answer 30", counts)
	}
	if counts.reasoning+counts.toolCalls+counts.answer != 50 {
		t.Errorf("phase counts sum to %d, want runner count 50", counts.reasoning+counts.toolCalls+counts.answer)
	}
}

func TestCountGeneratedTokenPhasesFallbackAndErrors(t *testing.T) {
	t.Run("no parsed content is attributed to answer", func(t *testing.T) {
		counts, err := countGeneratedTokenPhases(context.Background(), 9,
			func(context.Context, string) ([]int, error) {
				t.Fatal("tokenizer called without phase content")
				return nil, nil
			}, generatedTokenText{})
		if err != nil {
			t.Fatal(err)
		}
		if counts != (generatedTokenCounts{answer: 9}) {
			t.Errorf("counts = %+v, want all 9 tokens attributed to answer", counts)
		}
	})

	t.Run("tokenization errors are returned", func(t *testing.T) {
		text := generatedTokenText{}
		text.addReasoning("reason")
		wantErr := errors.New("tokenizer failed")
		_, err := countGeneratedTokenPhases(context.Background(), 1,
			func(context.Context, string) ([]int, error) { return nil, wantErr }, text)
		if !errors.Is(err, wantErr) {
			t.Errorf("error = %v, want %v", err, wantErr)
		}
	})

	t.Run("phase text is concatenated before tokenization", func(t *testing.T) {
		text := generatedTokenText{}
		text.addAnswer("answer")
		text.addAnswer(" continues")
		var got string
		counts, err := countGeneratedTokenPhases(context.Background(), 2,
			func(_ context.Context, content string) ([]int, error) {
				got = content
				return make([]int, len(strings.Fields(content))), nil
			}, text)
		if err != nil {
			t.Fatal(err)
		}
		if got != "answer continues" || counts.answer != 2 {
			t.Errorf("tokenized %q, counts %+v; want %q and two answer tokens", got, counts, "answer continues")
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
		parser := fakeSplitParser{thinkingTokens: 3, toolCallTokens: 2, answerTokens: 5}
		totalEval := time.Second
		answer, reasoning, toolCall := splitDurations(totalEval, parser)
		// reasoning = 1s * 3/10 = 300ms, toolCall = 1s * 2/10 = 200ms,
		// answer = 1s * 5/10 = 500ms.
		if reasoning != 300*time.Millisecond {
			t.Errorf("reasoning = %v, want 300ms", reasoning)
		}
		if toolCall != 200*time.Millisecond {
			t.Errorf("toolCall = %v, want 200ms", toolCall)
		}
		if answer != 500*time.Millisecond {
			t.Errorf("answer = %v, want 500ms", answer)
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
