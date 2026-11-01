package parsers

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type qwen3ParserState int

const (
	qwen3ParserStateLookingForThinkingOpen qwen3ParserState = iota
	qwen3ParserStateThinkingStartedEatingWhitespace
	qwen3ParserStateCollectingThinking
	qwen3ParserStateThinkingDoneEatingWhitespace
	qwen3ParserStateCollectingContent
	qwen3ParserStateToolStartedEatingWhitespace
	qwen3ParserStateCollectingToolContent
)

const (
	qwen3ThinkingOpenTag  = "<think>"
	qwen3ThinkingCloseTag = "</think>"
	qwen3ToolOpenTag      = "<tool_call>"
	qwen3ToolCloseTag     = "</tool_call>"
)

// Qwen3Parser parses Qwen3 output to extract thinking and tool calls.
// Qwen3 prompts end with <think> when thinking is enabled, so output begins
// with thinking content directly (without an opening tag).
type Qwen3Parser struct {
	state                  qwen3ParserState
	buffer                 strings.Builder
	tools                  []api.Tool
	callIndex              int
	hasThinkingSupport     bool
	defaultThinking        bool
	maybeThinkingOpenAtBOL bool
	// stateTransition holds the wall-clock time of the last parser state
	// transition. The zero value means no transition has happened yet.
	stateTransition time.Time
	// thinkingTokens counts the bytes of thinking content attributed to the
	// reasoning phase. It is used as the token-count proxy for [StateProvider].
	thinkingTokens int
	// toolCallTokens counts the bytes of tool-call content attributed to the
	// tool-calling phase. It is used as the token-count proxy for
	// [StateProvider].
	toolCallTokens int
	// answerTokens counts the bytes of answer content attributed to the
	// answer phase. It is used as the token-count proxy for [StateProvider].
	answerTokens int
}

// StateTransitionTime returns the wall-clock time at which the parser last
// transitioned to a different state, or the zero time if no transition has
// happened yet.
func (p *Qwen3Parser) StateTransitionTime() time.Time {
	return p.stateTransition
}

// ThinkingTokens returns the number of thinking-content bytes attributed to
// the reasoning phase, used as the token-count proxy for [StateProvider].
func (p *Qwen3Parser) ThinkingTokens() int { return p.thinkingTokens }

// ToolCallTokens returns the number of tool-call-content bytes attributed to
// the tool-calling phase, used as the token-count proxy for [StateProvider].
func (p *Qwen3Parser) ToolCallTokens() int { return p.toolCallTokens }

// AnswerTokens returns the number of answer-content bytes attributed to the
// answer phase, used as the token-count proxy for [StateProvider].
func (p *Qwen3Parser) AnswerTokens() int { return p.answerTokens }

// StateName returns the normalized name of the parser's current state:
// "thinking" while the thinking block is being collected, "thinkingDone"
// directly after the closing tag, "toolCalling" inside the tool-call region
// and "answer" otherwise.
func (p *Qwen3Parser) StateName() string {
	switch p.state {
	case qwen3ParserStateThinkingStartedEatingWhitespace, qwen3ParserStateCollectingThinking:
		return phaseThinking
	case qwen3ParserStateThinkingDoneEatingWhitespace:
		return phaseThinkingDone
	case qwen3ParserStateToolStartedEatingWhitespace, qwen3ParserStateCollectingToolContent:
		return phaseToolCalling
	default:
		return phaseAnswer
	}
}

// setState transitions the parser to next, recording the wall-clock time of
// the transition. No time is recorded when the state does not change.
func (p *Qwen3Parser) setState(next qwen3ParserState) {
	if p.state == next {
		return
	}
	p.stateTransition = time.Now()
	p.state = next
}

func (p *Qwen3Parser) HasToolSupport() bool {
	return true
}

func (p *Qwen3Parser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *Qwen3Parser) ThinkingClose() []string {
	if p.state == qwen3ParserStateCollectingThinking {
		return []string{qwen3ThinkingCloseTag}
	}
	return nil
}

func (p *Qwen3Parser) PreservedTokens() []string {
	return []string{
		qwen3ThinkingOpenTag,
		qwen3ThinkingCloseTag,
		qwen3ToolOpenTag,
		qwen3ToolCloseTag,
	}
}

func (p *Qwen3Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.buffer.Reset()
	p.callIndex = 0
	p.stateTransition = time.Time{}

	thinkingEnabled := thinkValue != nil && thinkValue.Bool()
	if thinkValue == nil {
		thinkingEnabled = p.defaultThinking
	}

	if p.hasThinkingSupport && thinkingEnabled {
		p.state = qwen3ParserStateCollectingThinking
		p.maybeThinkingOpenAtBOL = true
	} else {
		p.state = qwen3ParserStateCollectingContent
		p.maybeThinkingOpenAtBOL = false
	}
	return tools
}

type qwen3Event interface {
	isQwen3Event()
}

type qwen3EventContent struct {
	content string
}

func (qwen3EventContent) isQwen3Event() {}

type qwen3EventRawToolCall struct {
	raw string
}

func (qwen3EventRawToolCall) isQwen3Event() {}

type qwen3EventThinkingContent struct {
	content string
}

func (qwen3EventThinkingContent) isQwen3Event() {}

func (p *Qwen3Parser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case qwen3EventRawToolCall:
			p.toolCallTokens += len(event.raw)
			toolCall, err := parseQwen3ToolCall(event, p.tools)
			if err != nil {
				slog.Warn("qwen3 tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			toolCall.Function.Index = p.callIndex
			p.callIndex++
			calls = append(calls, toolCall)
		case qwen3EventThinkingContent:
			thinkingSb.WriteString(event.content)
			p.thinkingTokens += len(event.content)
		case qwen3EventContent:
			contentSb.WriteString(event.content)
			p.answerTokens += len(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), calls, nil
}

func (p *Qwen3Parser) parseEvents() []qwen3Event {
	var all []qwen3Event

	keepLooping := true
	for keepLooping {
		var events []qwen3Event
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "qwen3 events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

func (p *Qwen3Parser) eatLeadingWhitespaceAndTransitionTo(nextState qwen3ParserState) ([]qwen3Event, bool) {
	trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
	p.buffer.Reset()
	if trimmed == "" {
		return nil, false
	}
	p.setState(nextState)
	p.buffer.WriteString(trimmed)
	return nil, true
}

func (p *Qwen3Parser) splitAtTag(tag string, trimAfter bool) (string, string) {
	return splitAtTag(&p.buffer, tag, trimAfter)
}

func (p *Qwen3Parser) eat() ([]qwen3Event, bool) {
	var events []qwen3Event

	switch p.state {
	case qwen3ParserStateLookingForThinkingOpen:
		trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
		if strings.HasPrefix(trimmed, qwen3ThinkingOpenTag) {
			after := strings.TrimPrefix(trimmed, qwen3ThinkingOpenTag)
			after = strings.TrimLeftFunc(after, unicode.IsSpace)
			p.buffer.Reset()
			p.buffer.WriteString(after)
			if after == "" {
				p.setState(qwen3ParserStateThinkingStartedEatingWhitespace)
			} else {
				p.setState(qwen3ParserStateCollectingThinking)
			}
			return events, true
		} else if strings.HasPrefix(qwen3ThinkingOpenTag, trimmed) {
			return events, false
		} else if trimmed == "" {
			return events, false
		}
		p.setState(qwen3ParserStateCollectingContent)
		return events, true

	case qwen3ParserStateThinkingStartedEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(qwen3ParserStateCollectingThinking)

	case qwen3ParserStateCollectingThinking:
		acc := p.buffer.String()

		// Some qwen3 checkpoints emit an explicit opening <think> tag even
		// though the prompt already ended with <think>. Strip exactly one
		// leading opening tag if present.
		if p.maybeThinkingOpenAtBOL {
			trimmed := strings.TrimLeftFunc(acc, unicode.IsSpace)
			if strings.HasPrefix(trimmed, qwen3ThinkingOpenTag) {
				after := strings.TrimPrefix(trimmed, qwen3ThinkingOpenTag)
				after = strings.TrimLeftFunc(after, unicode.IsSpace)
				p.buffer.Reset()
				p.buffer.WriteString(after)
				if after == "" {
					return events, false
				}
				p.maybeThinkingOpenAtBOL = false
				return events, true
			}
			if strings.HasPrefix(qwen3ThinkingOpenTag, trimmed) {
				return events, false
			}
			p.maybeThinkingOpenAtBOL = false
		}

		thinkingCloseIdx := strings.Index(acc, qwen3ThinkingCloseTag)
		toolOpenIdx := strings.Index(acc, qwen3ToolOpenTag)

		// If a tool call starts before </think>, treat that as the end of thinking
		// for parsing purposes and continue in tool-call mode.
		if toolOpenIdx != -1 && (thinkingCloseIdx == -1 || toolOpenIdx < thinkingCloseIdx) {
			before, after := p.splitAtTag(qwen3ToolOpenTag, true)
			if len(before) > 0 {
				events = append(events, qwen3EventThinkingContent{content: before})
			}
			if after == "" {
				p.setState(qwen3ParserStateToolStartedEatingWhitespace)
			} else {
				p.setState(qwen3ParserStateCollectingToolContent)
			}
			return events, true
		}

		if strings.Contains(acc, qwen3ThinkingCloseTag) {
			thinking, remaining := p.splitAtTag(qwen3ThinkingCloseTag, true)
			if len(thinking) > 0 {
				events = append(events, qwen3EventThinkingContent{content: thinking})
			}
			if remaining == "" {
				p.setState(qwen3ParserStateThinkingDoneEatingWhitespace)
			} else {
				p.setState(qwen3ParserStateCollectingContent)
			}
			return events, true
		} else if overlapLen := max(overlap(acc, qwen3ThinkingCloseTag), overlap(acc, qwen3ToolOpenTag)); overlapLen > 0 {
			beforePartialTag := acc[:len(acc)-overlapLen]
			trailingWsLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWsLen

			unambiguous := acc[:ambiguousStart]
			ambiguous := acc[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwen3EventThinkingContent{content: unambiguous})
			}
			return events, false
		}

		whitespaceLen := trailingWhitespaceLen(acc)
		ambiguousStart := len(acc) - whitespaceLen
		unambiguous := acc[:ambiguousStart]
		ambiguous := acc[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, qwen3EventThinkingContent{content: unambiguous})
		}
		return events, false

	case qwen3ParserStateThinkingDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(qwen3ParserStateCollectingContent)

	case qwen3ParserStateCollectingContent:
		acc := p.buffer.String()
		if strings.Contains(acc, qwen3ToolOpenTag) {
			before, after := p.splitAtTag(qwen3ToolOpenTag, true)
			if len(before) > 0 {
				events = append(events, qwen3EventContent{content: before})
			}
			if after == "" {
				p.setState(qwen3ParserStateToolStartedEatingWhitespace)
			} else {
				p.setState(qwen3ParserStateCollectingToolContent)
			}
			return events, true
		} else if overlapLen := overlap(acc, qwen3ToolOpenTag); overlapLen > 0 {
			beforePartialTag := acc[:len(acc)-overlapLen]
			trailingWsLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWsLen

			unambiguous := acc[:ambiguousStart]
			ambiguous := acc[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwen3EventContent{content: unambiguous})
			}
			return events, false
		}

		whitespaceLen := trailingWhitespaceLen(acc)
		ambiguousStart := len(acc) - whitespaceLen
		unambiguous := acc[:ambiguousStart]
		ambiguous := acc[ambiguousStart:]
		p.buffer.Reset()
		p.buffer.WriteString(ambiguous)
		if len(unambiguous) > 0 {
			events = append(events, qwen3EventContent{content: unambiguous})
		}
		return events, false

	case qwen3ParserStateToolStartedEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(qwen3ParserStateCollectingToolContent)

	case qwen3ParserStateCollectingToolContent:
		acc := p.buffer.String()
		if strings.Contains(acc, qwen3ToolCloseTag) {
			toolContent, _ := p.splitAtTag(qwen3ToolCloseTag, true)
			if len(toolContent) == 0 {
				slog.Warn("qwen3 tool call closing tag found but no content before it")
			}
			events = append(events, qwen3EventRawToolCall{raw: toolContent})
			p.setState(qwen3ParserStateCollectingContent)
			return events, true
		}
		return events, false

	default:
		panic("unreachable")
	}
}

func parseQwen3ToolCall(raw qwen3EventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
	var parsed struct {
		Name      string                        `json:"name"`
		Arguments api.ToolCallFunctionArguments `json:"arguments"`
	}

	if err := json.Unmarshal([]byte(raw.raw), &parsed); err != nil {
		return api.ToolCall{}, fmt.Errorf("failed to parse JSON: %w", err)
	}

	if parsed.Name == "" {
		return api.ToolCall{}, fmt.Errorf("empty function name")
	}

	_ = tools // qwen3 uses direct JSON args and does not require schema coercion here.

	toolCall := api.ToolCall{
		Function: api.ToolCallFunction{
			Name:      parsed.Name,
			Arguments: parsed.Arguments,
		},
	}

	return toolCall, nil
}
