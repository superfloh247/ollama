package parsers

import (
	"context"
	"log/slog"
	"strings"
	"time"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
)

type olmo3ThinkParserState int

const (
	olmo3CollectingThink olmo3ThinkParserState = iota
	olmo3CollectingContent
)

const (
	olmo3ThinkCloseTag = "</think>"
)

type Olmo3ThinkParser struct {
	state  olmo3ThinkParserState
	buffer strings.Builder
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
func (p *Olmo3ThinkParser) StateTransitionTime() time.Time {
	return p.stateTransition
}

// ThinkingTokens returns the number of thinking-content bytes attributed to
// the reasoning phase, used as the token-count proxy for [StateProvider].
func (p *Olmo3ThinkParser) ThinkingTokens() int { return p.thinkingTokens }

// ToolCallTokens returns the number of tool-call-content bytes attributed to
// the tool-calling phase, used as the token-count proxy for [StateProvider].
func (p *Olmo3ThinkParser) ToolCallTokens() int { return p.toolCallTokens }

// AnswerTokens returns the number of answer-content bytes attributed to the
// answer phase, used as the token-count proxy for [StateProvider].
func (p *Olmo3ThinkParser) AnswerTokens() int { return p.answerTokens }

// StateName returns the normalized name of the parser's current state:
// "thinking" while the thinking block is being collected and "answer"
// otherwise.
func (p *Olmo3ThinkParser) StateName() string {
	if p.state == olmo3CollectingThink {
		return phaseThinking
	}
	return phaseAnswer
}

// setState transitions the parser to next, recording the wall-clock time of
// the transition. No time is recorded when the state does not change.
func (p *Olmo3ThinkParser) setState(next olmo3ThinkParserState) {
	if p.state == next {
		return
	}
	p.stateTransition = time.Now()
	p.state = next
}

func (p *Olmo3ThinkParser) HasToolSupport() bool {
	return false
}

func (p *Olmo3ThinkParser) HasThinkingSupport() bool {
	return true
}

func (p *Olmo3ThinkParser) ThinkingClose() []string {
	if p.state == olmo3CollectingThink {
		return []string{olmo3ThinkCloseTag}
	}
	return nil
}

func (p *Olmo3ThinkParser) PreservedTokens() []string {
	return []string{
		olmo3ThinkCloseTag,
	}
}

func (p *Olmo3ThinkParser) setInitialState(lastMessage *api.Message) {
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	// If prefilling with content, skip to content collection
	if prefill && lastMessage.Content != "" {
		p.state = olmo3CollectingContent
		return
	}

	// Model always thinks first (the <think> tag is injected in the prompt)
	p.state = olmo3CollectingThink
}

func (p *Olmo3ThinkParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.stateTransition = time.Time{}
	p.setInitialState(lastMessage)
	return tools
}

// Event types for internal parser communication
type olmo3Event interface {
	isOlmo3Event()
}

type olmo3EventThinkContent struct {
	content string
}

type olmo3EventContent struct {
	content string
}

func (olmo3EventThinkContent) isOlmo3Event() {}
func (olmo3EventContent) isOlmo3Event()      {}

func (p *Olmo3ThinkParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var contentSb strings.Builder
	var thinkingSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case olmo3EventThinkContent:
			thinkingSb.WriteString(event.content)
		case olmo3EventContent:
			contentSb.WriteString(event.content)
			p.answerTokens += len(event.content)
		}
	}

	return contentSb.String(), thinkingSb.String(), nil, nil
}

func (p *Olmo3ThinkParser) parseEvents() []olmo3Event {
	var all []olmo3Event

	keepLooping := true
	for keepLooping {
		var events []olmo3Event
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	if len(all) > 0 {
		slog.Log(context.TODO(), logutil.LevelTrace, "olmo3 events parsed", "events", all, "state", p.state, "buffer", p.buffer.String())
	}

	return all
}

func (p *Olmo3ThinkParser) eat() ([]olmo3Event, bool) {
	var events []olmo3Event
	bufStr := p.buffer.String()
	if bufStr == "" {
		return events, false
	}

	switch p.state {
	case olmo3CollectingThink:
		if strings.Contains(bufStr, olmo3ThinkCloseTag) {
			// Found complete </think> tag
			split := strings.SplitN(bufStr, olmo3ThinkCloseTag, 2)
			thinking := strings.TrimRightFunc(split[0], unicode.IsSpace)
			remaining := strings.TrimLeftFunc(split[1], unicode.IsSpace)

			p.buffer.Reset()
			p.buffer.WriteString(remaining)
			p.setState(olmo3CollectingContent)

			if len(thinking) > 0 {
				events = append(events, olmo3EventThinkContent{content: thinking})
			}
			return events, true
		} else if overlapLen := overlap(bufStr, olmo3ThinkCloseTag); overlapLen > 0 {
			// Partial </think> tag - withhold ambiguous content
			beforePartialTag := bufStr[:len(bufStr)-overlapLen]
			trailingLen := trailingWhitespaceLen(beforePartialTag)
			ambiguousStart := len(beforePartialTag) - trailingLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, olmo3EventThinkContent{content: unambiguous})
			}
			return events, false
		} else {
			// Regular thinking content - withhold trailing whitespace in case </think> follows
			whitespaceLen := trailingWhitespaceLen(bufStr)
			ambiguousStart := len(bufStr) - whitespaceLen

			unambiguous := bufStr[:ambiguousStart]
			ambiguous := bufStr[ambiguousStart:]
			p.buffer.Reset()
			p.buffer.WriteString(ambiguous)
			if len(unambiguous) > 0 {
				events = append(events, olmo3EventThinkContent{content: unambiguous})
			}
			return events, false
		}

	case olmo3CollectingContent:
		// Emit all content directly
		p.buffer.Reset()
		if len(bufStr) > 0 {
			events = append(events, olmo3EventContent{content: bufStr})
		}
		return events, false
	}

	return events, false
}
