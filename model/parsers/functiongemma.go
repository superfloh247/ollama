package parsers

import (
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

type FunctionGemmaParserState int

const (
	FunctionGemmaCollectingContent FunctionGemmaParserState = iota
	FunctionGemmaCollectingToolCalls
)

const (
	functionGemmaFunctionCallOpen  = "<start_function_call>"
	functionGemmaFunctionCallClose = "<end_function_call>"
)

// This format uses <start_function_call>call:name{args}<end_function_call> for tool calls.
type FunctionGemmaParser struct {
	state     FunctionGemmaParserState
	buffer    strings.Builder
	tools     []api.Tool
	callIndex int
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
func (p *FunctionGemmaParser) StateTransitionTime() time.Time {
	return p.stateTransition
}

// ThinkingTokens returns the number of thinking-content bytes attributed to
// the reasoning phase, used as the token-count proxy for [StateProvider].
func (p *FunctionGemmaParser) ThinkingTokens() int { return p.thinkingTokens }

// ToolCallTokens returns the number of tool-call-content bytes attributed to
// the tool-calling phase, used as the token-count proxy for [StateProvider].
func (p *FunctionGemmaParser) ToolCallTokens() int { return p.toolCallTokens }

// AnswerTokens returns the number of answer-content bytes attributed to the
// answer phase, used as the token-count proxy for [StateProvider].
func (p *FunctionGemmaParser) AnswerTokens() int { return p.answerTokens }

// StateName returns the normalized name of the parser's current state:
// "toolCalling" while function calls are being collected and "answer"
// otherwise.
func (p *FunctionGemmaParser) StateName() string {
	if p.state == FunctionGemmaCollectingToolCalls {
		return phaseToolCalling
	}
	return phaseAnswer
}

// setState transitions the parser to next, recording the wall-clock time of
// the transition. No time is recorded when the state does not change.
func (p *FunctionGemmaParser) setState(next FunctionGemmaParserState) {
	if p.state == next {
		return
	}
	p.stateTransition = time.Now()
	p.state = next
}

func (p *FunctionGemmaParser) HasToolSupport() bool     { return true }
func (p *FunctionGemmaParser) HasThinkingSupport() bool { return false }
func (p *FunctionGemmaParser) ThinkingClose() []string  { return nil }

func (p *FunctionGemmaParser) PreservedTokens() []string {
	return []string{
		functionGemmaFunctionCallOpen,
		functionGemmaFunctionCallClose,
	}
}

func (p *FunctionGemmaParser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.stateTransition = time.Time{}
	p.state = FunctionGemmaCollectingContent
	p.callIndex = 0
	return tools
}

type functionGemmaEvent interface {
	isFunctionGemmaEvent()
}

type FunctionGemmaEventContent struct {
	content string
}

type functionGemmaEventToolCall struct {
	toolCall api.ToolCall
}

func (FunctionGemmaEventContent) isFunctionGemmaEvent()  {}
func (functionGemmaEventToolCall) isFunctionGemmaEvent() {}

func (p *FunctionGemmaParser) Add(s string, done bool) (content string, thinking string, calls []api.ToolCall, err error) {
	p.buffer.WriteString(s)
	events := p.parseEvents()

	var toolCalls []api.ToolCall
	var contentSb strings.Builder
	for _, event := range events {
		switch event := event.(type) {
		case functionGemmaEventToolCall:
			toolCalls = append(toolCalls, event.toolCall)
		case FunctionGemmaEventContent:
			contentSb.WriteString(event.content)
			p.answerTokens += len(event.content)
		}
	}

	for i := range toolCalls {
		toolCalls[i].Function.Index = p.callIndex
		p.callIndex++
	}

	return contentSb.String(), "", toolCalls, nil
}

func (p *FunctionGemmaParser) parseEvents() []functionGemmaEvent {
	var all []functionGemmaEvent

	keepLooping := true
	for keepLooping {
		var events []functionGemmaEvent
		events, keepLooping = p.eat()
		if len(events) > 0 {
			all = append(all, events...)
		}
	}

	return all
}

// emitWithPartialCheck extracts unambiguous content before a potential partial tag
func (p *FunctionGemmaParser) emitWithPartialCheck(bufStr, tag string) (unambiguous, ambiguous string) {
	if overlapLen := overlap(bufStr, tag); overlapLen > 0 {
		beforePartialTag := bufStr[:len(bufStr)-overlapLen]
		return beforePartialTag, bufStr[len(beforePartialTag):]
	}
	return bufStr, ""
}

func (p *FunctionGemmaParser) eat() ([]functionGemmaEvent, bool) {
	bufStr := p.buffer.String()
	if bufStr == "" {
		return nil, false
	}

	switch p.state {
	case FunctionGemmaCollectingContent:
		if strings.Contains(bufStr, functionGemmaFunctionCallOpen) {
			split := strings.SplitN(bufStr, functionGemmaFunctionCallOpen, 2)
			content := split[0]
			p.buffer.Reset()
			p.buffer.WriteString(split[1])
			p.setState(FunctionGemmaCollectingToolCalls)
			if content != "" {
				return []functionGemmaEvent{FunctionGemmaEventContent{content: content}}, true
			}
			return nil, true
		}
		unambig, ambig := p.emitWithPartialCheck(bufStr, functionGemmaFunctionCallOpen)
		p.buffer.Reset()
		p.buffer.WriteString(ambig)
		if unambig != "" {
			return []functionGemmaEvent{FunctionGemmaEventContent{content: unambig}}, false
		}
		return nil, false

	case FunctionGemmaCollectingToolCalls:
		if strings.Contains(bufStr, functionGemmaFunctionCallClose) {
			split := strings.SplitN(bufStr, functionGemmaFunctionCallClose, 2)
			remaining := split[1]
			p.buffer.Reset()
			p.buffer.WriteString(remaining)

			var events []functionGemmaEvent
			if tc, err := p.parseToolCall(split[0]); err == nil {
				events = append(events, functionGemmaEventToolCall{toolCall: tc})
			}

			if !strings.Contains(remaining, functionGemmaFunctionCallOpen) {
				p.setState(FunctionGemmaCollectingContent)
			}
			return events, true
		}
		return nil, false
	}

	return nil, false
}

// Matches call:function_name{args}
var functionGemmaCallRegex = regexp.MustCompile(`call:([^{]+)\{(.*)\}`)

func (p *FunctionGemmaParser) parseToolCall(content string) (api.ToolCall, error) {
	toolCall := api.ToolCall{}

	// Extract function name and arguments
	match := functionGemmaCallRegex.FindStringSubmatch(content)
	if len(match) < 3 {
		return toolCall, nil
	}

	toolCall.Function.Name = match[1]
	argsStr := match[2]

	// Parse arguments
	toolCall.Function.Arguments = p.parseArguments(argsStr)

	return toolCall, nil
}

// parseArguments parses the key:value,key:value format
func (p *FunctionGemmaParser) parseArguments(argsStr string) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	if argsStr == "" {
		return args
	}

	// Split by comma, but handle nested structures
	parts := p.splitArguments(argsStr)

	for _, part := range parts {
		// Find the first colon to split key:value
		colonIdx := strings.Index(part, ":")
		if colonIdx == -1 {
			continue
		}

		key := part[:colonIdx]
		value := part[colonIdx+1:]

		// Parse the value
		args.Set(key, p.parseValue(value))
	}

	return args
}

// splitArguments splits arguments by comma, respecting nested structures
func (p *FunctionGemmaParser) splitArguments(argsStr string) []string {
	var parts []string
	var current strings.Builder
	depth := 0
	inEscape := false

	for i := 0; i < len(argsStr); i++ {
		ch := argsStr[i]

		// Check for <escape> tags
		if i+8 <= len(argsStr) && argsStr[i:i+8] == "<escape>" {
			inEscape = !inEscape
			current.WriteString("<escape>")
			i += 7 // Skip the rest of <escape>
			continue
		}

		if !inEscape {
			switch ch {
			case '{', '[':
				depth++
				current.WriteByte(ch)
			case '}', ']':
				depth--
				current.WriteByte(ch)
			case ',':
				if depth == 0 {
					if current.Len() > 0 {
						parts = append(parts, current.String())
						current.Reset()
					}
					continue
				}
				current.WriteByte(ch)
			default:
				current.WriteByte(ch)
			}
		} else {
			current.WriteByte(ch)
		}
	}

	if current.Len() > 0 {
		parts = append(parts, current.String())
	}

	return parts
}

// parseValue parses a single value from the FunctionGemma format
func (p *FunctionGemmaParser) parseValue(value string) any {
	// Check for escaped string
	if strings.HasPrefix(value, "<escape>") && strings.HasSuffix(value, "<escape>") {
		// Remove the escape tags
		return value[8 : len(value)-8]
	}

	// Check for boolean
	if value == "true" {
		return true
	}
	if value == "false" {
		return false
	}

	// Check for number
	if num, ok := parseNumber(value); ok {
		return num
	}

	// Check for array
	if strings.HasPrefix(value, "[") && strings.HasSuffix(value, "]") {
		return p.parseArray(value[1 : len(value)-1])
	}

	// Check for object
	if strings.HasPrefix(value, "{") && strings.HasSuffix(value, "}") {
		return p.parseObject(value[1 : len(value)-1])
	}

	// Default to string
	return value
}

// parseArray parses an array value
func (p *FunctionGemmaParser) parseArray(content string) []any {
	var result []any
	parts := p.splitArguments(content)
	for _, part := range parts {
		result = append(result, p.parseValue(part))
	}
	return result
}

// parseObject parses an object value
func (p *FunctionGemmaParser) parseObject(content string) map[string]any {
	result := make(map[string]any)
	parts := p.splitArguments(content)
	for _, part := range parts {
		colonIdx := strings.Index(part, ":")
		if colonIdx == -1 {
			continue
		}
		key := part[:colonIdx]
		value := part[colonIdx+1:]
		result[key] = p.parseValue(value)
	}
	return result
}

// parseNumber tries to parse a string as a number
func parseNumber(s string) (any, bool) {
	// Try integer first
	var intVal int64
	if _, err := fmt.Sscanf(s, "%d", &intVal); err == nil {
		// Check if the entire string was consumed
		if fmt.Sprintf("%d", intVal) == s {
			return intVal, true
		}
	}

	// Try float
	var floatVal float64
	if _, err := fmt.Sscanf(s, "%f", &floatVal); err == nil {
		return floatVal, true
	}

	return nil, false
}
