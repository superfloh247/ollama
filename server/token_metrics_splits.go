package server

import (
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/parsers"
	"github.com/ollama/ollama/thinking"
	"github.com/ollama/ollama/tools"
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
	total     int
	complete  bool
}

func newGeneratedTokenCounts(enabled bool) *generatedTokenCounts {
	if !enabled {
		return nil
	}
	return &generatedTokenCounts{complete: true}
}

func (c *generatedTokenCounts) addToken(phase string) {
	if c == nil {
		return
	}
	switch phase {
	case phaseStateThinking, phaseStateThinkingDone:
		c.reasoning++
	case phaseStateToolCalling:
		c.toolCalls++
	case phaseStateAnswer:
		c.answer++
	default:
		c.complete = false
	}
	c.total++
}

func tokenPhase(before, after string) string {
	if !knownTokenPhase(before) || !knownTokenPhase(after) {
		return ""
	}
	if before == phaseStateThinking {
		return phaseStateThinking
	}
	if before == phaseStateToolCalling || after == phaseStateToolCalling {
		return phaseStateToolCalling
	}
	if after == phaseStateThinking {
		return phaseStateThinking
	}
	return phaseStateAnswer
}

func knownTokenPhase(phase string) bool {
	switch phase {
	case phaseStateThinking, phaseStateThinkingDone, phaseStateToolCalling, phaseStateAnswer:
		return true
	default:
		return false
	}
}

func parserPhase(parser any) string {
	if p, ok := parser.(interface{ StateName() string }); ok {
		return p.StateName()
	}
	return ""
}

func generatedTokensMatch(content string, tokens []llm.Logprob) bool {
	if len(tokens) == 0 {
		return content == ""
	}
	var reconstructed strings.Builder
	for _, token := range tokens {
		reconstructed.WriteString(token.Token)
	}
	return reconstructed.String() == content
}

func addParserTokens(parser parsers.Parser, content string, done bool, tokens []llm.Logprob, counts *generatedTokenCounts, clock *PhaseClock) (parsedContent, parsedThinking string, calls []api.ToolCall, err error) {
	if counts == nil {
		return parser.Add(content, done)
	}
	if !generatedTokensMatch(content, tokens) {
		counts.complete = false
		return parser.Add(content, done)
	}
	if len(tokens) == 0 {
		return parser.Add(content, done)
	}
	var contentBuilder, thinkingBuilder strings.Builder
	for i, token := range tokens {
		before := parserPhase(parser)
		content, thinking, parsedCalls, err := parser.Add(token.Token, done && i == len(tokens)-1)
		if err != nil {
			return "", "", nil, err
		}
		notePhaseClock(clock, parser)
		counts.addToken(tokenPhase(before, parserPhase(parser)))
		contentBuilder.WriteString(content)
		thinkingBuilder.WriteString(thinking)
		calls = append(calls, parsedCalls...)
	}
	return contentBuilder.String(), thinkingBuilder.String(), calls, nil
}

func addThinkingTokens(parser *thinking.Parser, content string, done bool, tokens []llm.Logprob, counts *generatedTokenCounts, clock *PhaseClock) (reasoning, answer string) {
	if counts == nil {
		return parser.AddContent(content)
	}
	if !generatedTokensMatch(content, tokens) {
		counts.complete = false
		return parser.AddContent(content)
	}
	if len(tokens) == 0 {
		return parser.AddContent(content)
	}
	var reasoningBuilder, answerBuilder strings.Builder
	for _, token := range tokens {
		before := parser.StateName()
		thought, remaining := parser.AddContent(token.Token)
		notePhaseClock(clock, parser)
		counts.addToken(tokenPhase(before, parser.StateName()))
		reasoningBuilder.WriteString(thought)
		answerBuilder.WriteString(remaining)
	}
	if done {
		thought, remaining := parser.AddContent("")
		reasoningBuilder.WriteString(thought)
		answerBuilder.WriteString(remaining)
	}
	return reasoningBuilder.String(), answerBuilder.String()
}

func addThinkingAndToolTokens(
	thinkingParser *thinking.Parser,
	toolParser *tools.Parser,
	content string,
	done bool,
	tokens []llm.Logprob,
	counts *generatedTokenCounts,
	clock *PhaseClock,
) (reasoning, answer string, calls []api.ToolCall) {
	if counts == nil {
		reasoning, answer = thinkingParser.AddContent(content)
		calls, answer = toolParser.Add(answer)
		return
	}
	if !generatedTokensMatch(content, tokens) {
		counts.complete = false
		reasoning, answer = thinkingParser.AddContent(content)
		calls, answer = toolParser.Add(answer)
		return
	}
	if len(tokens) == 0 {
		reasoning, answer = thinkingParser.AddContent(content)
		calls, answer = toolParser.Add(answer)
		return
	}

	var reasoningBuilder, answerBuilder strings.Builder
	for i, token := range tokens {
		thinkingBefore := thinkingParser.StateName()
		toolBefore := toolParser.StateName()
		thought, remaining := thinkingParser.AddContent(token.Token)
		notePhaseClock(clock, thinkingParser)
		var parsedCalls []api.ToolCall
		var parsedContent string
		if remaining != "" {
			parsedCalls, parsedContent = toolParser.Add(remaining)
			notePhaseClock(clock, toolParser)
		}
		thinkingAfter := thinkingParser.StateName()
		toolAfter := toolParser.StateName()
		phase := tokenPhase(thinkingBefore, thinkingAfter)
		if thinkingBefore != phaseStateThinking && thinkingBefore != phaseStateThinkingDone &&
			thinkingAfter != phaseStateThinking && thinkingAfter != phaseStateThinkingDone {
			phase = tokenPhase(toolBefore, toolAfter)
		}
		counts.addToken(phase)
		reasoningBuilder.WriteString(thought)
		answerBuilder.WriteString(parsedContent)
		calls = append(calls, parsedCalls...)
		if done && i == len(tokens)-1 {
			thought, remaining = thinkingParser.AddContent("")
			reasoningBuilder.WriteString(thought)
			if remaining != "" {
				parsedCalls, parsedContent = toolParser.Add(remaining)
				calls = append(calls, parsedCalls...)
				answerBuilder.WriteString(parsedContent)
			}
		}
	}
	return reasoningBuilder.String(), answerBuilder.String(), calls
}

func addToolTokens(parser *tools.Parser, content string, tokens []llm.Logprob, counts *generatedTokenCounts, clock *PhaseClock) (calls []api.ToolCall, parsedContent string) {
	if counts == nil {
		return parser.Add(content)
	}
	if !generatedTokensMatch(content, tokens) {
		counts.complete = false
		return parser.Add(content)
	}
	if len(tokens) == 0 {
		return parser.Add(content)
	}
	var contentBuilder strings.Builder
	for _, token := range tokens {
		before := parser.StateName()
		parsedCalls, content := parser.Add(token.Token)
		notePhaseClock(clock, parser)
		counts.addToken(tokenPhase(before, parser.StateName()))
		calls = append(calls, parsedCalls...)
		contentBuilder.WriteString(content)
	}
	return calls, contentBuilder.String()
}

func addAnswerTokens(content string, tokens []llm.Logprob, counts *generatedTokenCounts) {
	if counts == nil {
		return
	}
	if !generatedTokensMatch(content, tokens) {
		counts.complete = false
		return
	}
	for range tokens {
		counts.addToken(phaseStateAnswer)
	}
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
