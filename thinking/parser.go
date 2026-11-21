package thinking

import (
	"strings"
	"time"
	"unicode"
)

type thinkingState int

// Normalized phase names exposed via [Parser.StateName]; they must match the
// phase vocabulary of the server package's PhaseClock.
const (
	phaseThinking     = "thinking"
	phaseThinkingDone = "thinkingDone"
	phaseToolCalling  = "toolCalling"
	phaseAnswer       = "answer"
)

const (
	// We're looking for the opening tag, but we haven't seen any non-whitespace
	// characters yet
	thinkingState_LookingForOpening thinkingState = iota
	// We've seen the opening tag, but we haven't seen any non-whitespace
	// characters yet (we want to eat any whitespace between the opening tag and
	// the thinking content)
	thinkingState_ThinkingStartedEatingWhitespace
	// We've seen non-whitespace characters after the opening tag, but we haven't
	// seen the closing tag yet
	thinkingState_Thinking
	// We've seen the closing tag, but we haven't seen any non-whitespace
	// characters after the closing tag yet (we want to eat any whitespace between
	// the closing tag and the content)
	thinkingState_ThinkingDoneEatingWhitespace
	// We've seen the closing tag and seen at least one non-whitespace character
	// after it
	thinkingState_ThinkingDone
)

func (s thinkingState) String() string {
	switch s {
	case thinkingState_LookingForOpening:
		return "LookingForOpening"
	case thinkingState_ThinkingStartedEatingWhitespace:
		return "ThinkingStartedEatingWhitespace"
	case thinkingState_Thinking:
		return "Thinking"
	case thinkingState_ThinkingDoneEatingWhitespace:
		return "ThinkingDoneEatingWhitespace"
	case thinkingState_ThinkingDone:
		return "ThinkingDone"
	default:
		return "Unknown"
	}
}

type Parser struct {
	state      thinkingState
	OpeningTag string
	ClosingTag string
	acc        strings.Builder
	// stateTransition holds the wall-clock time of the last parser state
	// transition. The zero value means no transition has happened yet.
	stateTransition time.Time
	// thinkingTokens counts the bytes of thinking content attributed to the
	// reasoning phase. It is used as the token-count proxy for [StateProvider].
	thinkingTokens int
	// answerTokens counts the bytes of answer content attributed to the answer
	// phase. It is used as the token-count proxy for [StateProvider].
	answerTokens int
}

// StateTransitionTime returns the wall-clock time at which the parser last
// transitioned to a different state, or the zero time if no transition has
// happened yet.
func (s *Parser) StateTransitionTime() time.Time {
	return s.stateTransition
}

// StateName returns the normalized name of the parser's current state:
// "thinking" while thinking content is being collected, "thinkingDone" once
// the closing tag has been seen, and "answer" otherwise.
func (s *Parser) StateName() string {
	switch s.state {
	case thinkingState_ThinkingStartedEatingWhitespace, thinkingState_Thinking:
		return phaseThinking
	case thinkingState_ThinkingDoneEatingWhitespace, thinkingState_ThinkingDone:
		return phaseThinkingDone
	default:
		return phaseAnswer
	}
}

// ThinkingTokens returns the number of thinking-content bytes attributed to
// the reasoning phase, used as the token-count proxy for [StateProvider].
func (s *Parser) ThinkingTokens() int { return s.thinkingTokens }

// AnswerTokens returns the number of answer-content bytes attributed to the
// answer phase, used as the token-count proxy for [StateProvider].
func (s *Parser) AnswerTokens() int { return s.answerTokens }

// ToolCallTokens returns 0: the thinking parser does not handle tool calls.
func (s *Parser) ToolCallTokens() int { return 0 }

// setState transitions the parser to next, recording the wall-clock time of
// the transition. No time is recorded when the state does not change.
func (s *Parser) setState(next thinkingState) {
	if s.state == next {
		return
	}
	s.stateTransition = time.Now()
	s.state = next
}

// AddContent returns the thinking content and the non-thinking content that
// should be immediately sent to the user. It will internally buffer if it needs
// to see more raw content to disambiguate
func (s *Parser) AddContent(content string) (string, string) {
	s.acc.WriteString(content)

	var thinkingSb, remainingSb strings.Builder

	var thinking, remaining string
	keepLooping := true
	// we loop because we might pass through multiple parsing states in a single
	// call to addContent, and we want to make sure callers don't have to wait for
	// data that's already unambiguous
	for keepLooping {
		thinking, remaining, keepLooping = eat(s)
		thinkingSb.WriteString(thinking)
		remainingSb.WriteString(remaining)
	}

	return thinkingSb.String(), remainingSb.String()
}

// the additional bool return is true iff we should continue eating
func eat(s *Parser) (string, string, bool) {
	switch s.state {
	case thinkingState_LookingForOpening:
		trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)
		if strings.HasPrefix(trimmed, s.OpeningTag) {
			after := strings.Join(strings.Split(trimmed, s.OpeningTag)[1:], s.OpeningTag)
			after = strings.TrimLeftFunc(after, unicode.IsSpace)
			// after might contain more than just thinking tokens, so we continue
			// parsing instead of returning it as thinking tokens here
			s.acc.Reset()
			s.acc.WriteString(after)
			if after == "" {
				s.setState(thinkingState_ThinkingStartedEatingWhitespace)
			} else {
				s.setState(thinkingState_Thinking)
			}
			return "", "", true
		} else if strings.HasPrefix(s.OpeningTag, trimmed) {
			// partial opening seen, so let's keep accumulating
			return "", "", false
		} else if trimmed == "" {
			// saw whitespace only, so let's keep accumulating
			return "", "", false
		} else {
			// didn't see an opening tag, but we have content, so thinking was skipped
			s.setState(thinkingState_ThinkingDone)
			// note that we use the original content, not the trimmed one because we
			// don't want to eat any whitespace in the real content if there were no
			// thinking tags
			untrimmed := s.acc.String()
			s.acc.Reset()
			return "", untrimmed, false
		}
	case thinkingState_ThinkingStartedEatingWhitespace:
		trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)
		s.acc.Reset()
		if trimmed == "" {
			return "", "", false
		} else {
			s.setState(thinkingState_Thinking)
			s.acc.WriteString(trimmed)
			return "", "", true
		}
	case thinkingState_Thinking:
		acc := s.acc.String()
		if strings.Contains(acc, s.ClosingTag) {
			split := strings.Split(acc, s.ClosingTag)
			thinking := split[0]
			remaining := strings.Join(split[1:], s.ClosingTag)
			remaining = strings.TrimLeftFunc(remaining, unicode.IsSpace)
			s.acc.Reset()
			if remaining == "" {
				s.setState(thinkingState_ThinkingDoneEatingWhitespace)
			} else {
				s.setState(thinkingState_ThinkingDone)
			}
			s.thinkingTokens += len(thinking)
			return thinking, remaining, false
		} else if overlapLen := overlap(acc, s.ClosingTag); overlapLen > 0 {
			thinking := acc[:len(acc)-overlapLen]
			remaining := acc[len(acc)-overlapLen:]
			s.acc.Reset()
			// keep track of the candidate closing tag. We have to buffer it until it
			// becomes disambiguated
			s.acc.WriteString(remaining)
			s.thinkingTokens += len(thinking)
			return thinking, "", false
		} else {
			// purely just thinking tokens, so we can return them
			s.acc.Reset()
			s.thinkingTokens += len(acc)
			return acc, "", false
		}
	case thinkingState_ThinkingDoneEatingWhitespace:
		trimmed := strings.TrimLeftFunc(s.acc.String(), unicode.IsSpace)
		s.acc.Reset()
		// if we see non-whitespace, we're done eating the leading whitespace of the content
		if trimmed != "" {
			s.setState(thinkingState_ThinkingDone)
			s.answerTokens += len(trimmed)
		}
		return "", trimmed, false
	case thinkingState_ThinkingDone:
		acc := s.acc.String()
		s.acc.Reset()
		s.answerTokens += len(acc)
		return "", acc, false
	default:
		panic("unknown state")
	}
}

// longest overlap between suffix of s and prefix of delim
func overlap(s, delim string) int {
	max := min(len(delim), len(s))
	for i := max; i > 0; i-- {
		if strings.HasSuffix(s, delim[:i]) {
			return i
		}
	}
	return 0
}
