package parsers

import (
	"testing"

	"github.com/ollama/ollama/api"
)

// StateProvider exposes the per-phase token counts a streaming parser has
// attributed to reasoning (e.g. inside <tool_call>) and to tool calls while parsing the
// model output.
type StateProvider interface {
	ThinkingTokens() int
	ToolCallTokens() int
}

// TestStateProviderThinkingTokens verifies that parsers with thinking support
// correctly report thinking token counts via StateProvider.
func TestStateProviderThinkingTokens(t *testing.T) {
	cases := []struct {
		parserName string
		input      string
	}{
		{"qwen3.5", "<think>Let me think about this.</think>The answer is 42."},
		{"deepseek3", "<think>Let me think about this.</think>The answer is 42."},
	}

	for _, tc := range cases {
		t.Run(tc.parserName, func(t *testing.T) {
			parser := ParserForName(tc.parserName)
			if parser == nil {
				t.Fatalf("parser %q not found", tc.parserName)
			}

			thinkValue := &api.ThinkValue{Value: true}
			parser.Init(nil, nil, thinkValue)

			sp, ok := parser.(StateProvider)
			if !ok {
				t.Fatalf("parser %q does not implement StateProvider", tc.parserName)
			}

			_, _, _, err := parser.Add(tc.input, true)
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}

			if sp.ThinkingTokens() == 0 {
				t.Errorf("expected ThinkingTokens() > 0 for %q, got 0", tc.parserName)
			}
		})
	}
}

// TestStateProviderToolCallTokens verifies that parsers with tool support
// correctly report tool-call token counts via StateProvider.
func TestStateProviderToolCallTokens(t *testing.T) {
	cases := []struct {
		parserName string
		input      string
	}{
		{"qwen3.5", "<tool_call><function=get_weather><parameter=location>\nSan Francisco\n</parameter></function></tool_call>"},
		{"qwen3-coder", "<tool_call><function=get_weather><parameter=location>\nSan Francisco\n</parameter></function></tool_call>"},
	}

	for _, tc := range cases {
		t.Run(tc.parserName, func(t *testing.T) {
			parser := ParserForName(tc.parserName)
			if parser == nil {
				t.Fatalf("parser %q not found", tc.parserName)
			}

			thinkValue := &api.ThinkValue{Value: false}
			parser.Init(nil, nil, thinkValue)

			sp, ok := parser.(StateProvider)
			if !ok {
				t.Fatalf("parser %q does not implement StateProvider", tc.parserName)
			}

			_, _, _, err := parser.Add(tc.input, true)
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}

			if sp.ToolCallTokens() == 0 {
				t.Errorf("expected ToolCallTokens() > 0 for %q, got 0", tc.parserName)
			}
		})
	}
}
