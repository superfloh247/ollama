package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestQwen35ParserXMLToolCall(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						props.Set("days", api.ToolProperty{Type: api.PropertyType{"integer"}})
						return props
					}(),
				},
			},
		},
	}

	parser.Init(tools, nil, &api.ThinkValue{Value: false})
	input := "<tool_call><function=get_weather><parameter=location>\nSan Francisco\n</parameter><parameter=days>\n3\n</parameter></function></tool_call>"
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}

	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}

	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "San Francisco" {
		t.Fatalf("expected location %q, got %v", "San Francisco", location)
	}

	days, ok := calls[0].Function.Arguments.Get("days")
	if !ok || days != 3 {
		t.Fatalf("expected days %d, got %v", 3, days)
	}
}

func TestQwen35ParserToolCallByteCountsAreIncremental(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init([]api.Tool{{Function: api.ToolFunction{Name: "weather"}}}, nil, &api.ThinkValue{Value: false})
	state, ok := parser.(interface{ ToolCallTokens() int })
	if !ok {
		t.Fatal("expected qwen3.5 parser to report tool-call byte counts")
	}

	firstCall := "<tool_call><function=weather><parameter=city>\nParis\n</parameter></function></tool_call>"
	secondCall := "<tool_call><function=weather><parameter=city>\nLondon\n</parameter></function></tool_call>"
	firstCallBytes := len(firstCall) - len("<tool_call>") - len("</tool_call>")
	secondCallBytes := len(secondCall) - len("<tool_call>") - len("</tool_call>")

	for _, chunk := range []struct {
		text string
		want int
	}{
		{firstCall, firstCallBytes},
		{secondCall, firstCallBytes + secondCallBytes},
		{"Answer.", firstCallBytes + secondCallBytes},
	} {
		if _, _, _, err := parser.Add(chunk.text, false); err != nil {
			t.Fatalf("parse chunk %q: %v", chunk.text, err)
		}
		if got := state.ToolCallTokens(); got != chunk.want {
			t.Errorf("ToolCallTokens() after %q = %d, want %d", chunk.text, got, chunk.want)
		}
	}
}

func TestQwen35ParserThinkingWithExplicitOpeningTag(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("<think>\nLet me think...</think>Answer.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "Let me think..." {
		t.Fatalf("expected thinking %q, got %q", "Let me think...", thinking)
	}
	if content != "Answer." {
		t.Fatalf("expected content %q, got %q", "Answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserAssistantPrefillStartsInContent(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	last := &api.Message{Role: "assistant", Content: "Prefilled response start"}
	parser.Init(nil, last, nil)

	content, thinking, calls, err := parser.Add(" and continued", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected no thinking for assistant prefill continuation, got %q", thinking)
	}
	if content != " and continued" {
		t.Fatalf("expected content %q, got %q", " and continued", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserToolCallEmittedInThinkingIsParsed(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}

	parser.Init(tools, nil, &api.ThinkValue{Value: true})
	input := `Need weather lookup<tool_call><function=get_weather><parameter=location>
SF
</parameter></function></tool_call>`
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "Need weather lookup" {
		t.Fatalf("expected thinking %q, got %q", "Need weather lookup", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}

	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}

	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "SF" {
		t.Fatalf("expected location %q, got %v", "SF", location)
	}
}

func TestQwen35ParserToolCallEmittedInThinkingIsParsedWhenToolCallTagIsSplitAcrossChunks(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}

	parser.Init(tools, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("Need weather lookup<tool_c", false)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "Need weather lookup" {
		t.Fatalf("expected thinking %q, got %q", "Need weather lookup", thinking)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls in first chunk, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add(`all><function=get_weather><parameter=location>
SF
</parameter></function></tool_call>`, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "" {
		t.Fatalf("expected no additional thinking, got %q", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}

	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}

	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "SF" {
		t.Fatalf("expected location %q, got %v", "SF", location)
	}
}

func TestQwen35ParserFakeoutPartialToolCallThenThinkCloseAcrossChunks(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}

	parser.Init(tools, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("Need weather lookup<tool_c", false)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "Need weather lookup" {
		t.Fatalf("expected thinking %q, got %q", "Need weather lookup", thinking)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls in first chunk, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add("</thi", false)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "<tool_c" {
		t.Fatalf("expected thinking %q, got %q", "<tool_c", thinking)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls in second chunk, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add("nk>", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "" {
		t.Fatalf("expected no additional thinking in third chunk, got %q", thinking)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls in third chunk, got %d", len(calls))
	}
}

func TestQwen35ParserToolCallAfterThinkingCloseIsParsed(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	tools := []api.Tool{
		{
			Function: api.ToolFunction{
				Name: "get_weather",
				Parameters: api.ToolFunctionParameters{
					Properties: func() *api.ToolPropertiesMap {
						props := api.NewToolPropertiesMap()
						props.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
						return props
					}(),
				},
			},
		},
	}

	parser.Init(tools, nil, &api.ThinkValue{Value: true})
	input := `Need weather lookup</think><tool_call><function=get_weather><parameter=location>
SF
</parameter></function></tool_call>`
	content, thinking, calls, err := parser.Add(input, true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if thinking != "Need weather lookup" {
		t.Fatalf("expected thinking %q, got %q", "Need weather lookup", thinking)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call after </think>, got %d", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("expected tool name %q, got %q", "get_weather", calls[0].Function.Name)
	}

	location, ok := calls[0].Function.Arguments.Get("location")
	if !ok || location != "SF" {
		t.Fatalf("expected location %q, got %v", "SF", location)
	}
}

func TestQwen35ParserThinkingDisabledPassesContentThrough(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: false})
	content, thinking, calls, err := parser.Add("Plain answer without think close tag.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != "Plain answer without think close tag." {
		t.Fatalf("expected content %q, got %q", "Plain answer without think close tag.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserThinkingDisabledWithCloseTagTreatsAsContent(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: false})
	content, thinking, calls, err := parser.Add("</think>Some content after spurious tag.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != "</think>Some content after spurious tag." {
		t.Fatalf("expected content %q, got %q", "</think>Some content after spurious tag.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserLeadingThinkCloseProducesContent(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("</think>The final answer.", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "" {
		t.Fatalf("expected empty thinking, got %q", thinking)
	}
	if content != "The final answer." {
		t.Fatalf("expected content %q, got %q", "The final answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserStreamingSplitThinkCloseTag(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add("Reasoning text</thi", false)
	if err != nil {
		t.Fatalf("parse failed on first chunk: %v", err)
	}
	if thinking != "Reasoning text" {
		t.Fatalf("expected thinking %q, got %q", "Reasoning text", thinking)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add("nk>The final answer.", true)
	if err != nil {
		t.Fatalf("parse failed on second chunk: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected no additional thinking on second chunk, got %q", thinking)
	}
	if content != "The final answer." {
		t.Fatalf("expected content %q, got %q", "The final answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserStreamingEatsWhitespaceAfterThinkClose(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})

	content, thinking, calls, err := parser.Add("Reasoning</think>", false)
	if err != nil {
		t.Fatalf("parse failed on first chunk: %v", err)
	}
	if thinking != "Reasoning" {
		t.Fatalf("expected thinking %q, got %q", "Reasoning", thinking)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add("\n \t", false)
	if err != nil {
		t.Fatalf("parse failed on whitespace chunk: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected no thinking on whitespace chunk, got %q", thinking)
	}
	if content != "" {
		t.Fatalf("expected whitespace after </think> to be eaten, got content %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}

	content, thinking, calls, err = parser.Add("The final answer.", true)
	if err != nil {
		t.Fatalf("parse failed on content chunk: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected no additional thinking, got %q", thinking)
	}
	if content != "The final answer." {
		t.Fatalf("expected content %q, got %q", "The final answer.", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserThinkingTruncatedWithoutCloseTag(t *testing.T) {
	parser := ParserForName("qwen3.5")
	if parser == nil {
		t.Fatal("expected qwen3.5 parser")
	}

	parser.Init(nil, nil, &api.ThinkValue{Value: true})
	content, thinking, calls, err := parser.Add("Reasoning that never closes", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}

	if thinking != "Reasoning that never closes" {
		t.Fatalf("expected thinking %q, got %q", "Reasoning that never closes", thinking)
	}
	if content != "" {
		t.Fatalf("expected empty content, got %q", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
}

func TestQwen35ParserStateTransitionTime(t *testing.T) {
	// note: tags are built from parts to avoid literal tool tags in this file
	const (
		thinkOpen  = "<" + "think>"
		thinkClose = "</" + "think>"
	)

	p := &Qwen35Parser{}
	p.Init(nil, nil, &api.ThinkValue{Value: true})

	if !p.StateTransitionTime().IsZero() {
		t.Fatalf("expected zero StateTransitionTime before any transition, got %v", p.StateTransitionTime())
	}

	// thinking content without the close tag keeps the parser in the thinking
	// state, so no transition must be recorded
	_, thinking, calls, err := p.Add("Let me think about this.", false)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "Let me think about this." {
		t.Fatalf("expected thinking %q, got %q", "Let me think about this.", thinking)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
	if !p.StateTransitionTime().IsZero() {
		t.Fatalf("expected zero StateTransitionTime while still collecting thinking, got %v", p.StateTransitionTime())
	}

	// closing the thinking block transitions the parser into content collection
	content, thinking, calls, err := p.Add(thinkClose+"The answer", true)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if thinking != "" {
		t.Fatalf("expected no thinking, got %q", thinking)
	}
	if content != "The answer" {
		t.Fatalf("expected content %q, got %q", "The answer", content)
	}
	if len(calls) != 0 {
		t.Fatalf("expected no tool calls, got %d", len(calls))
	}
	if p.StateTransitionTime().IsZero() {
		t.Fatal("expected non-zero StateTransitionTime after the thinking phase ended")
	}
}
