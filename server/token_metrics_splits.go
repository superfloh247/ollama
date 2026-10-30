package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// StateProvider exposes parser-measured per-phase content sizes used to
// estimate timing splits. These are byte counts, not generated-token counts.
type StateProvider interface {
	// ThinkingTokens returns the number of reasoning-content bytes.
	ThinkingTokens() int
	// ToolCallTokens returns the number of tool-call-content bytes.
	ToolCallTokens() int
}

// answerTokenProvider is optionally implemented by parsers that track answer
// content bytes in addition to reasoning and tool-call bytes.
type answerTokenProvider interface {
	AnswerTokens() int
}

type generatedTokenCounts struct {
	reasoning int
	toolCalls int
	answer    int
}

type generatedTokenText struct {
	reasoning strings.Builder
	toolCalls strings.Builder
	answer    strings.Builder
}

func newGeneratedTokenText(enabled bool) *generatedTokenText {
	if !enabled {
		return nil
	}
	return &generatedTokenText{}
}

func (g *generatedTokenText) addReasoning(s string) {
	if g != nil {
		g.reasoning.WriteString(s)
	}
}

func (g *generatedTokenText) addAnswer(s string) {
	if g != nil {
		g.answer.WriteString(s)
	}
}

func (g *generatedTokenText) addToolCalls(calls []api.ToolCall) error {
	if g == nil {
		return nil
	}
	for _, call := range calls {
		args, err := json.Marshal(call.Function.Arguments)
		if err != nil {
			return fmt.Errorf("marshal tool-call arguments: %w", err)
		}
		fmt.Fprintf(&g.toolCalls, "<tool_call><function=%s>\n%s\n</function></tool_call>", call.Function.Name, args)
	}
	return nil
}

func countGeneratedTokenPhases(
	ctx context.Context,
	evalCount int,
	tokenize func(context.Context, string) ([]int, error),
	text generatedTokenText,
) (generatedTokenCounts, error) {
	var counts generatedTokenCounts
	if evalCount <= 0 {
		return counts, nil
	}
	if tokenize == nil {
		return counts, errors.New("tokenizer is required to count generated token phases")
	}

	for _, phase := range []struct {
		content string
		count   *int
	}{
		{text.reasoning.String(), &counts.reasoning},
		{text.toolCalls.String(), &counts.toolCalls},
		{text.answer.String(), &counts.answer},
	} {
		if phase.content == "" {
			continue
		}
		tokens, err := tokenize(ctx, phase.content)
		if err != nil {
			return generatedTokenCounts{}, err
		}
		*phase.count = len(tokens)
	}

	total := counts.reasoning + counts.toolCalls + counts.answer
	if total == 0 {
		counts.answer = evalCount
		return counts, nil
	}

	counts.reasoning = evalCount * counts.reasoning / total
	counts.toolCalls = evalCount * counts.toolCalls / total
	counts.answer = evalCount - counts.reasoning - counts.toolCalls
	return counts, nil
}

// splitDurations splits the total evaluation duration into the reasoning,
// tool-call and answer parts. The parser's byte counts provide the phase
// proportions. The three returned values
// always sum to totalEval.
//
// When the parser is nil (no parser / no parser support) or no content has
// been attributed to any phase, all of totalEval is returned as answer duration.
func splitDurations(totalEval time.Duration, parser StateProvider) (answer, reasoning, toolCall time.Duration) {
	if parser == nil {
		answer = totalEval
		return
	}

	thinking := parser.ThinkingTokens()
	if thinking < 0 {
		thinking = 0
	}
	toolCalls := parser.ToolCallTokens()
	if toolCalls < 0 {
		toolCalls = 0
	}
	answerCount := 0
	if ap, ok := parser.(answerTokenProvider); ok {
		answerCount = ap.AnswerTokens()
		if answerCount < 0 {
			answerCount = 0
		}
	}
	// If no content is attributed to any phase, everything goes to answer.
	if thinking+toolCalls+answerCount == 0 {
		answer = totalEval
		return
	}

	// Split totalEval proportionally based on the token distribution.
	// We clamp the split so that the sum never exceeds totalEval.
	totalAttributed := thinking + toolCalls + answerCount

	// Calculate durations proportional to per-phase content counts.
	reasoning = time.Duration(float64(totalEval) * float64(thinking) / float64(totalAttributed))
	toolCall = time.Duration(float64(totalEval) * float64(toolCalls) / float64(totalAttributed))
	answer = totalEval - reasoning - toolCall

	// Clamp negative values (can happen due to rounding) to 0.
	if answer < 0 {
		answer = 0
	}
	return
}
