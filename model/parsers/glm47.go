package parsers

import "github.com/ollama/ollama/api"

// GLM47Parser extends GLM46Parser with thinking-aware initialization.
// GLM-4.7's prompt ends with <think> when thinking is enabled, so the parser
// must start in CollectingThinking state (the model outputs thinking content directly).
type GLM47Parser struct {
	GLM46Parser
}

func (p *GLM47Parser) Init(tools []api.Tool, lastMessage *api.Message, thinkValue *api.ThinkValue) []api.Tool {
	p.tools = tools
	p.callIndex = 0
	p.thinkingEnabled = thinkValue == nil || thinkValue.Bool()
	// When thinking is enabled (nil or true), the prompt ends with <think>,
	// so model output starts directly with thinking content (no opening tag).
	if p.thinkingEnabled {
		p.state = glm46ParserState_CollectingThinking
	}
	return tools
}

// ThinkingTokens returns the number of thinking-content bytes attributed to
// the reasoning phase, used as the token-count proxy for [StateProvider].
func (p *GLM47Parser) ThinkingTokens() int { return p.thinkingTokens }

// ToolCallTokens returns the number of tool-call-content bytes attributed to
// the tool-calling phase, used as the token-count proxy for [StateProvider].
func (p *GLM47Parser) ToolCallTokens() int { return p.toolCallTokens }

// AnswerTokens returns the number of answer-content bytes attributed to the
// answer phase, used as the token-count proxy for [StateProvider].
func (p *GLM47Parser) AnswerTokens() int { return p.answerTokens }
