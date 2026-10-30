package parsers

import (
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"time"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

// TODO: call the init function
const (
	CollectingThinkingContent qwenParserState = iota
	CollectingContent
	CollectingToolContent
	ThinkingDoneEatingWhitespace
	ToolCallDoneEatingWhitespace
)

const (
	thinkingCloseTag = "</think>"
)

type Qwen3VLParser struct {
	state              qwenParserState
	buffer             strings.Builder
	tools              []api.Tool
	callIndex          int
	hasThinkingSupport bool
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
func (p *Qwen3VLParser) StateTransitionTime() time.Time {
	return p.stateTransition
}

// StateName returns the normalized name of the parser's current state:
// "thinking" while the thinking block is being collected, "thinkingDone"
// directly after the closing tag, "toolCalling" in the tool-call region and
// "answer" otherwise.
func (p *Qwen3VLParser) StateName() string {
	switch p.state {
	case CollectingThinkingContent:
		return phaseThinking
	case ThinkingDoneEatingWhitespace:
		return phaseThinkingDone
	case CollectingToolContent, ToolCallDoneEatingWhitespace:
		return phaseToolCalling
	default:
		return phaseAnswer
	}
}

// setState transitions the parser to next, recording the wall-clock time of
// the transition. No time is recorded when the state does not change.
func (p *Qwen3VLParser) setState(next qwenParserState) {
	if p.state == next {
		return
	}
	p.stateTransition = time.Now()
	p.state = next
}

func (p *Qwen3VLParser) HasToolSupport() bool {
	return true
}

func (p *Qwen3VLParser) HasThinkingSupport() bool {
	return p.hasThinkingSupport
}

func (p *Qwen3VLParser) ThinkingClose() []string {
	if p.state == CollectingThinkingContent {
		return []string{thinkingCloseTag}
	}
	return nil
}

func (p *Qwen3VLParser) PreservedTokens() []string {
	return []string{
		thinkingCloseTag,
		toolOpenTag,
		toolCloseTag,
	}
}

func (p *Qwen3VLParser) setInitialState(lastMessage *api.Message) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"
	if !p.HasThinkingSupport() {
		p.state = CollectingContent
		return
	}

	if prefill && lastMessage.Content != "" {
		p.state = CollectingContent
		return
	}

	p.state = CollectingThinkingContent
}

func (p *Qwen3VLParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0
	p.stateTransition = time.Time{}
	p.setInitialState(lastMessage)
	return tools
}

type qwenEventThinkingContent struct {
	content string
}

func (qwenEventThinkingContent) isQwenEvent() {}

func (p *Qwen3VLParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case qwenEventRawToolCall:
			toolCall, err := parseJSONToolCall(event, p.tools)
			if err != nil {
				slog.Warn("qwen tool call parsing failed", "error", err)
				return "", "", nil, err
			}
			calls = append(calls, toolCall)
		case qwenEventThinkingContent:
			thinkingSb.WriteString(event.content)
		case qwenEventContent:
			// TODO(drifkin): if the same turn contains multiple interleaved content
			// events, we naively append them together here.
			contentSb.WriteString(event.content)
			p.answerTokens += len(event.content)
		}
	}

	for i := range calls {
		calls[i].Function.Index = p.callIndex
		p.callIndex++
	}

	return contentSb.String(), thinkingSb.String(), calls, nil
}

func (p *Qwen3VLParser) parseEvents() []qwenEvent {
	var all []qwenEvent

	keepLooping := true
	for keepLooping {
		var events []qwenEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "qwen events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

func (p *Qwen3VLParser) eatLeadingWhitespaceAndTransitionTo(nextState qwenParserState) ([]qwenEvent, bool) {
	trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
	p.buffer.Reset()
	if trimmed == "" {
		return nil, false
	}
	p.setState(nextState)
	p.buffer.WriteString(trimmed)
	return nil, true
}

func (p *Qwen3VLParser) eat() ([]qwenEvent, bool) {
	var events []qwenEvent

	switch p.state {
	case CollectingContent:
		if strings.Contains(p.buffer.String(), toolOpenTag) {
			// events = emitContentBeforeTag(p, events, toolOpenTag)
			before, _ := splitAtTag(&p.buffer, toolOpenTag, false)
			if len(before) > 0 {
				events = append(events, qwenEventContent{content: before})
			}
			p.setState(CollectingToolContent)
			return events, true
		} else if overlapLen := overlap(p.buffer.String(), toolOpenTag); overlapLen > 0 {
			beforePartialTag := p.buffer.String()[:len(p.buffer.String())-overlapLen]
			trailingWhitespaceLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventContent{content: unambiguous})
			}
			return events, false
		} else {
			whitespaceLen := trailingWhitespaceLen(p.buffer.String())
			ambiguousStart := len(p.buffer.String()) - whitespaceLen

			unambiguous := p.buffer.String()[:ambiguousStart]
			ambiguous := p.buffer.String()[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventContent{content: unambiguous})
			}
			return events, false
		}
	case CollectingToolContent:
		if strings.Contains(p.buffer.String(), toolCloseTag) {
			split := strings.SplitN(p.buffer.String(), toolCloseTag, 2)
			before := split[0] // do we also need to do it to tool calls?
			if len(before) == 0 {
				slog.Warn("qwen tool call closing tag found but no content before it")
			}

			after := split[1]
			events = append(events, qwenEventRawToolCall{raw: before})
			p.buffer.Reset()
			p.buffer.WriteString(after)
			p.setState(ToolCallDoneEatingWhitespace)
			return events, true
		} else {
			return events, false
		}
	case CollectingThinkingContent:
		acc := p.buffer.String()
		thinkingCloseIdx := strings.Index(acc, thinkingCloseTag)
		toolOpenIdx := strings.Index(acc, toolOpenTag)

		// If a tool call starts before </think>, treat that as the end of thinking
		// for parsing purposes and continue in tool-call mode.
		if toolOpenIdx != -1 && (thinkingCloseIdx == -1 || toolOpenIdx < thinkingCloseIdx) {
			before, _ := splitAtTag(&p.buffer, toolOpenTag, false)
			if len(before) > 0 {
				events = append(events, qwenEventThinkingContent{content: before})
			}
			p.setState(CollectingToolContent)
			return events, true
		}

		if strings.Contains(acc, thinkingCloseTag) {
			thinking, remaining := splitAtTag(&p.buffer, thinkingCloseTag, true)
			if len(thinking) > 0 {
				events = append(events, qwenEventThinkingContent{content: thinking})
			}
			if remaining == "" {
				p.setState(ThinkingDoneEatingWhitespace)
			} else {
				p.setState(CollectingContent)
			}
			return events, true
		} else if overlapLen := max(overlap(acc, thinkingCloseTag), overlap(acc, toolOpenTag)); overlapLen > 0 {
			beforePartialTag := acc[:len(acc)-overlapLen]
			trailingWhitespaceLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen

			unambiguous := acc[:ambiguousStart]
			ambiguous := acc[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventThinkingContent{content: unambiguous})
			}
			return events, false
		} else {
			whitespaceLen := trailingWhitespaceLen(acc)
			ambiguousStart := len(acc) - whitespaceLen

			unambiguous := acc[:ambiguousStart]
			ambiguous := acc[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, qwenEventThinkingContent{content: unambiguous})
			}
			return events, false
		}
	case ThinkingDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(CollectingContent)
	case ToolCallDoneEatingWhitespace:
		return p.eatLeadingWhitespaceAndTransitionTo(CollectingContent)
	default:
		panic("unreachable")
	}
}

func parseJSONToolCall(raw qwenEventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
	var toolCallFunction api.ToolCallFunction
	if err := json.Unmarshal([]byte(raw.raw), &toolCallFunction); err != nil {
		return api.ToolCall{}, err
	}

	toolCall := api.ToolCall{}
	toolCall.Function = toolCallFunction

	return toolCall, nil
}
