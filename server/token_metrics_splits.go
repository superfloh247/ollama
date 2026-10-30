package server

import (
	"time"
)

// StateProvider exposes the per-phase token counts a streaming parser has
// attributed to reasoning (e.g. inside <tool_call>) and to tool calls while parsing the
// model output. Parsers that do not track token counts (or models without
// parser support) are not passed — [splitGeneratedTokens] treats a nil
// StateProvider as "no split information available".
type StateProvider interface {
	// ThinkingTokens returns how many tokens of the generated output the parser
	// classified as reasoning.
	ThinkingTokens() int
	// ToolCallTokens returns how many tokens of the generated output the parser
	// classified as tool calls.
	ToolCallTokens() int
}

// splitGeneratedTokens splits the total number of generated tokens (m.EvalCount)
// into the reasoning, tool-call and answer parts. The parser state provides
// the reasoning and tool-call counts; the answer is the remainder. When the
// parser is nil (no parser / no parser support), or the parser counts are
// missing or inconsistent (e.g. exceed evalCount), everything that cannot be
// attributed to reasoning or tool calls is counted as answer. The three
// returned values always satisfy
//
//	reasoning + toolCalls + answer == evalCount
//
// with reasoning, toolCalls and answer each >= 0.
// answerTokenProvider is optionally implemented by parsers that track answer
// (content) byte counts in addition to reasoning and tool-call counts.
type answerTokenProvider interface {
	AnswerTokens() int
}

// splitGeneratedTokens splits the total number of generated tokens (m.EvalCount)
// into the reasoning, tool-call and answer parts. The parser state provides
// the per-phase byte counts; tokens are allocated proportionally to those
// counts so the three parts sum exactly to evalCount. When the parser is nil
// (no parser / no parser support), everything is attributed to the answer.
func splitGeneratedTokens(evalCount int, parser StateProvider) (reasoning, toolCalls, answer int) {
	if parser == nil {
		answer = evalCount
		return
	}

	reasoningBytes := parser.ThinkingTokens()
	if reasoningBytes < 0 {
		reasoningBytes = 0
	}
	toolBytes := parser.ToolCallTokens()
	if toolBytes < 0 {
		toolBytes = 0
	}
	answerBytes := 0
	if ap, ok := parser.(answerTokenProvider); ok {
		answerBytes = ap.AnswerTokens()
		if answerBytes < 0 {
			answerBytes = 0
		}
	}

	totalBytes := reasoningBytes + toolBytes + answerBytes
	if answerBytes > 0 && totalBytes > 0 {
		reasoning = evalCount * reasoningBytes / totalBytes
		toolCalls = evalCount * toolBytes / totalBytes
	} else {
		// Legacy behaviour: treat the reasoning/tool-call counts as token
		// counts and clamp them to the total.
		reasoning = reasoningBytes
		toolCalls = toolBytes
		if over := reasoning + toolCalls - evalCount; over > 0 {
			if over <= toolCalls {
				toolCalls -= over
			} else {
				reasoning -= over - toolCalls
				toolCalls = 0
			}
			if reasoning < 0 {
				reasoning = 0
			}
		}
	}
	answer = evalCount - reasoning - toolCalls
	if answer < 0 {
		answer = 0
	}
	return
}

// splitDurations splits the total evaluation duration into the reasoning,
// tool-call and answer parts. The parser state provides the reasoning and
// tool-call token counts; the time for each phase is proportional to the
// fraction of tokens attributed to that phase. The three returned values
// always sum to totalEval.
//
// When the parser is nil (no parser / no parser support) or no tokens have
// been attributed to reasoning or tool calls, all of totalEval is returned
// as answer duration.
func splitDurations(totalEval time.Duration, parser StateProvider) (answer, reasoning, toolCall time.Duration) {
	var toolCalls int
	if parser == nil {
		answer = totalEval
		return
	}

	thinking := parser.ThinkingTokens()
	if thinking < 0 {
		thinking = 0
	}
	toolCalls = parser.ToolCallTokens()
	if toolCalls < 0 {
		toolCalls = 0
	}
	// If no tokens are attributed to any phase, everything goes to answer.
	if thinking+toolCalls == 0 {
		answer = totalEval
		return
	}

	// Split totalEval proportionally based on the token distribution.
	// We clamp the split so that the sum never exceeds totalEval.
	totalAttributed := thinking + toolCalls
	if totalAttributed == 0 {
		answer = totalEval
		return
	}

	// Calculate durations proportional to token counts, then clamp to totalEval.
	reasoning = time.Duration(float64(totalEval) * float64(thinking) / float64(totalAttributed))
	toolCall = time.Duration(float64(totalEval) * float64(toolCalls) / float64(totalAttributed))
	answer = totalEval - reasoning - toolCall

	// Clamp negative values (can happen due to rounding) to 0.
	if answer < 0 {
		answer = 0
	}
	return
}
