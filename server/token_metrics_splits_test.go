package server

import (
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/parsers"
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

func TestAddParserTokensCountsExactPhaseTokens(t *testing.T) {
	parser := parsers.ParserForName("qwen3.5")
	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	tokens := []llm.Logprob{
		{TokenLogprob: llm.TokenLogprob{Token: "reason"}},
		{TokenLogprob: llm.TokenLogprob{Token: "</think>"}},
		{TokenLogprob: llm.TokenLogprob{Token: "answer"}},
	}
	counts := newGeneratedTokenCounts(true)

	content, thinking, _, err := addParserTokens(parser, "reason</think>answer", true, tokens, counts, nil)
	if err != nil {
		t.Fatal(err)
	}
	if content != "answer" || thinking != "reason" {
		t.Errorf("parsed content/thinking = %q/%q, want %q/%q", content, thinking, "answer", "reason")
	}
	if counts.reasoning != 2 || counts.toolCalls != 0 || counts.answer != 1 || counts.total != 3 || !counts.complete {
		t.Errorf("counts = %+v, want exactly 2 reasoning and 1 answer token", counts)
	}
}

func TestAddParserTokensCountsToolCallTokens(t *testing.T) {
	parser := parsers.ParserForName("qwen3-coder")
	parser.Init([]api.Tool{{Function: api.ToolFunction{Name: "weather"}}}, nil, nil)
	tokens := []llm.Logprob{
		{TokenLogprob: llm.TokenLogprob{Token: "hello"}},
		{TokenLogprob: llm.TokenLogprob{Token: "<tool_call>\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "<function=weather>\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "<parameter=city>\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "Paris\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "</parameter>\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "</function>\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "</tool_call>"}},
		{TokenLogprob: llm.TokenLogprob{Token: "done"}},
	}
	counts := newGeneratedTokenCounts(true)

	content, _, calls, err := addParserTokens(parser, "hello<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>done", true, tokens, counts, nil)
	if err != nil {
		t.Fatal(err)
	}
	if content != "hellodone" || len(calls) != 1 || calls[0].Function.Name != "weather" {
		t.Fatalf("parsed content/calls = %q/%+v, want %q and one weather call", content, calls, "hellodone")
	}
	if counts.reasoning != 0 || counts.toolCalls != 7 || counts.answer != 2 || counts.total != 9 || !counts.complete {
		t.Errorf("counts = %+v, want 7 tool-call and 2 answer tokens", counts)
	}
}

func TestAddQwen35ParserTokensCountsToolCallTokens(t *testing.T) {
	parser := parsers.ParserForName("qwen3.5")
	parser.Init([]api.Tool{{Function: api.ToolFunction{Name: "weather"}}}, nil, &api.ThinkValue{Value: false})
	tokens := []llm.Logprob{
		{TokenLogprob: llm.TokenLogprob{Token: "hello"}},
		{TokenLogprob: llm.TokenLogprob{Token: "<tool_call>\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "<function=weather>\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "<parameter=city>\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "Paris\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "</parameter>\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "</function>\n"}},
		{TokenLogprob: llm.TokenLogprob{Token: "</tool_call>"}},
		{TokenLogprob: llm.TokenLogprob{Token: "done"}},
	}
	counts := newGeneratedTokenCounts(true)

	content, _, calls, err := addParserTokens(parser, "hello<tool_call>\n<function=weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>done", true, tokens, counts, nil)
	if err != nil {
		t.Fatal(err)
	}
	if content != "hellodone" || len(calls) != 1 || calls[0].Function.Name != "weather" {
		t.Fatalf("parsed content/calls = %q/%+v, want %q and one weather call", content, calls, "hellodone")
	}
	if counts.reasoning != 0 || counts.toolCalls != 7 || counts.answer != 2 || counts.total != 9 || !counts.complete {
		t.Errorf("counts = %+v, want 7 tool-call and 2 answer tokens", counts)
	}
}

func TestAddParserTokensRejectsUnalignedTokenData(t *testing.T) {
	parser := parsers.ParserForName("passthrough")
	parser.Init(nil, nil, nil)
	counts := newGeneratedTokenCounts(true)

	content, _, _, err := addParserTokens(parser, "actual output", true, []llm.Logprob{
		{TokenLogprob: llm.TokenLogprob{Token: "different"}},
	}, counts, nil)
	if err != nil {
		t.Fatal(err)
	}
	if content != "actual output" {
		t.Errorf("content = %q, want raw response unchanged", content)
	}
	if counts.complete {
		t.Fatal("expected accounting to be incomplete when token text does not match response content")
	}
}

func TestGeneratedTokenCountsRejectUnknownPhases(t *testing.T) {
	counts := newGeneratedTokenCounts(true)
	counts.addToken("")

	if counts.complete || counts.total != 1 {
		t.Fatalf("counts = %+v, want incomplete accounting with one observed token", counts)
	}
	if counts.reasoning != 0 || counts.toolCalls != 0 || counts.answer != 0 {
		t.Fatalf("unknown phase was attributed to a known phase: %+v", counts)
	}
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
