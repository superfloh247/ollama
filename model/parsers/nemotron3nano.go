package parsers

import (
	"strings"
	"time"
	"unicode"

	"github.com/ollama/ollama/api"
)

type Nemotron3NanoParserState int

const (
	Nemotron3NanoCollectingThinking Nemotron3NanoParserState = iota
	Nemotron3NanoSkipWhitespaceAfterThinking
	Nemotron3NanoCollectingContent
)

const (
	nemotronThinkOpen    = "<think>"
	nemotronThinkClose   = "</think>"
	nemotronToolCallOpen = "<tool_call>"
)

type Nemotron3NanoParser struct {
	state                  Nemotron3NanoParserState
	buffer                 strings.Builder
	toolParser             *Qwen3CoderParser
	maybeThinkingOpenAtBOL bool
	skipThinkingLeadingWS  bool
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
func (p *Nemotron3NanoParser) StateTransitionTime() time.Time {
	if p.toolParser != nil {
		if transition := p.toolParser.StateTransitionTime(); transition.After(p.stateTransition) {
			return transition
		}
	}
	return p.stateTransition
}

func (p *Nemotron3NanoParser) StateName() string {
	if p.toolParser != nil && p.toolParser.StateName() == phaseToolCalling {
		return phaseToolCalling
	}
	switch p.state {
	case Nemotron3NanoCollectingThinking:
		return phaseThinking
	case Nemotron3NanoSkipWhitespaceAfterThinking:
		return phaseThinkingDone
	default:
		return phaseAnswer
	}
}

// ThinkingTokens returns the number of thinking-content bytes attributed to
// the reasoning phase, used as the token-count proxy for [StateProvider].
func (p *Nemotron3NanoParser) ThinkingTokens() int { return p.thinkingTokens }

// ToolCallTokens returns the number of tool-call-content bytes attributed to
// the tool-calling phase, used as the token-count proxy for [StateProvider].
// setState transitions the parser to next, recording the wall-clock time of
// the transition. No time is recorded when the state does not change.
func (p *Nemotron3NanoParser) setState(next Nemotron3NanoParserState) {
	if p.state == next {
		return
	}
	p.stateTransition = time.Now()
	p.state = next
}

func (p *Nemotron3NanoParser) HasToolSupport() bool     { return true }
func (p *Nemotron3NanoParser) HasThinkingSupport() bool { return true }

func (p *Nemotron3NanoParser) ThinkingClose() []string {
	if p.state == Nemotron3NanoCollectingThinking {
		return []string{nemotronThinkClose}
	}
	return nil
}

func (p *Nemotron3NanoParser) PreservedTokens() []string {
	return []string{
		nemotronThinkOpen,
		nemotronThinkClose,
		toolOpenTag,
		toolCloseTag,
	}
}

func (p *Nemotron3NanoParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.toolParser = &Qwen3CoderParser{}
	p.toolParser.Init(tools, nil, nil)
	p.buffer.Reset()
	p.maybeThinkingOpenAtBOL = false
	p.skipThinkingLeadingWS = false
	p.stateTransition = time.Time{}

	thinkingEnabled := thinkValue == nil || thinkValue.Bool()
	prefill := lastMessage != nil && lastMessage.Role == "assistant"

	if !thinkingEnabled || (prefill && lastMessage.Content != "") {
		p.state = Nemotron3NanoCollectingContent
	} else {
		p.state = Nemotron3NanoCollectingThinking
		p.maybeThinkingOpenAtBOL = true
	}

	return tools
}

func (p *Nemotron3NanoParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	if p.state == Nemotron3NanoCollectingContent {
		return p.toolParser.Add(s, done)
	}

	if p.state == Nemotron3NanoSkipWhitespaceAfterThinking {
		s = strings.TrimLeftFunc(s, unicode.IsSpace)
		if s == "" {
			return "", "", nil, nil
		}
		p.setState(Nemotron3NanoCollectingContent)
		return p.toolParser.Add(s, done)
	}

	// Nemotron3NanoCollectingThinking - buffer and look for end markers
	p.buffer.WriteString(s)
	if p.skipThinkingLeadingWS {
		trimmed := strings.TrimLeftFunc(p.buffer.String(), unicode.IsSpace)
		p.buffer.Reset()
		p.buffer.WriteString(trimmed)
		if trimmed == "" {
			return "", "", nil, nil
		}
		p.skipThinkingLeadingWS = false
	}
	if p.stripOpeningThinkTag() {
		return p.Add("", done)
	}
	if p.maybeThinkingOpenAtBOL {
		bufStr := p.buffer.String()
		trimmed := strings.TrimLeftFunc(bufStr, unicode.IsSpace)
		if trimmed == "" || overlap(trimmed, nemotronThinkOpen) == len(trimmed) {
			if len(trimmed) != len(bufStr) {
				p.buffer.Reset()
				p.buffer.WriteString(trimmed)
			}
			if done {
				thinking = p.buffer.String()
				p.buffer.Reset()
				p.maybeThinkingOpenAtBOL = false
				return "", thinking, nil, nil
			}
			return "", "", nil, nil
		}
	}
	bufStr := p.buffer.String()

	// Look for end of thinking: </think> or <tool_call> (model may skip </think>)
	thinkIdx := strings.Index(bufStr, nemotronThinkClose)
	toolIdx := strings.Index(bufStr, nemotronToolCallOpen)

	var endIdx int = -1
	var remainder string

	if thinkIdx != -1 && (toolIdx == -1 || thinkIdx < toolIdx) {
		endIdx = thinkIdx
		remainder = strings.TrimLeftFunc(bufStr[thinkIdx+len(nemotronThinkClose):], unicode.IsSpace)
	} else if toolIdx != -1 {
		endIdx = toolIdx
		remainder = bufStr[toolIdx:] // Include <tool_call> tag
	}

	if endIdx != -1 {
		thinking = strings.TrimRightFunc(bufStr[:endIdx], unicode.IsSpace)
		p.buffer.Reset()

		if remainder == "" {
			p.setState(Nemotron3NanoSkipWhitespaceAfterThinking)
		} else {
			p.setState(Nemotron3NanoCollectingContent)
			content, _, calls, err = p.toolParser.Add(remainder, done)
		}
		return content, thinking, calls, err
	}

	// No end marker - emit unambiguous thinking
	thinking = p.emitThinking(bufStr)
	if done {
		thinking += p.buffer.String()
		p.buffer.Reset()
	}
	return "", thinking, nil, nil
}

// emitThinking returns unambiguous thinking content, keeping potential partial tags in buffer
func (p *Nemotron3NanoParser) emitThinking(bufStr string) string {
	// Check for partial </think> or <tool_call> at end
	thinkOverlap := overlap(bufStr, nemotronThinkClose)
	toolOverlap := overlap(bufStr, nemotronToolCallOpen)
	maxOverlap := max(thinkOverlap, toolOverlap)

	if maxOverlap > 0 {
		beforePartialTag := bufStr[:len(bufStr)-maxOverlap]
		ambiguousStart := len(beforePartialTag) - trailingWhitespaceLen(beforePartialTag)
		unambiguous := bufStr[:ambiguousStart]
		p.buffer.Reset()
		p.buffer.WriteString(bufStr[ambiguousStart:])
		return unambiguous
	}

	// No partial tags - emit all but trailing whitespace
	wsLen := trailingWhitespaceLen(bufStr)
	if wsLen > 0 {
		unambiguous := bufStr[:len(bufStr)-wsLen]
		p.buffer.Reset()
		p.buffer.WriteString(bufStr[len(bufStr)-wsLen:])
		return unambiguous
	}

	// Nothing to hold back
	p.buffer.Reset()
	return bufStr
}

func (p *Nemotron3NanoParser) stripOpeningThinkTag() bool {
	if !p.maybeThinkingOpenAtBOL {
		return false
	}

	bufStr := p.buffer.String()
	trimmed := strings.TrimLeftFunc(bufStr, unicode.IsSpace)
	if trimmed == "" {
		p.buffer.Reset()
		return false
	}

	if strings.HasPrefix(trimmed, nemotronThinkOpen) {
		p.buffer.Reset()
		p.buffer.WriteString(strings.TrimLeftFunc(trimmed[len(nemotronThinkOpen):], unicode.IsSpace))
		p.maybeThinkingOpenAtBOL = false
		p.skipThinkingLeadingWS = true
		return true
	}

	if overlap(trimmed, nemotronThinkOpen) == len(trimmed) {
		if len(trimmed) != len(bufStr) {
			p.buffer.Reset()
			p.buffer.WriteString(trimmed)
		}
		return false
	}

	p.maybeThinkingOpenAtBOL = false
	return false
}
