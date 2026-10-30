package server

import (
	"reflect"
	"sync"
	"time"
)

// The normalized phase names a [StateTransitioner] reports via StateName.
// [PhaseClock.Note] attributes each transition to a generation phase.
const (
	phaseStateThinking     = "thinking"
	phaseStateThinkingDone = "thinkingDone"
	phaseStateToolCalling  = "toolCalling"
	phaseStateAnswer       = "answer"
)

// StateTransitioner is implemented by parsers that track the generation
// phase (reasoning / tool calling / answer) in a state machine. It exposes
// the wall-clock time of the last state transition together with
// the normalized name of the state the parser is currently in. Parsers
// without a state machine (e.g. the passthrough parser) do not implement it.
type StateTransitioner interface {
	// StateTransitionTime returns the wall-clock time at which the parser
	// last transitioned to a different state, or the zero time if no
	// transition has happened yet.
	StateTransitionTime() time.Time
	// StateName returns the normalized name of the state the parser is
	// currently in: "thinking", "thinkingDone", "toolCalling" or "answer".
	StateName() string
}

// PhaseClock accumulates wall-clock durations for each generation phase. It is
// created per request and fed from the parser or structured response callback.
// The clock is safe for concurrent use.
type PhaseClock struct {
	mu                sync.Mutex
	startTime         time.Time
	thinkStart        time.Time
	toolCallStart     time.Time
	answerStart       time.Time
	lastTransition    time.Time
	activePhase       string
	reasoningDuration time.Duration
	toolCallDuration  time.Duration
}

// Begin records the wall-clock time of the first stream chunk. It is a no-op
// after the first call and safe to invoke on every chunk.
func (pc *PhaseClock) Begin() {
	if pc == nil {
		return
	}
	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.startTime.IsZero() {
		pc.startTime = time.Now()
	}
}

// Note records parser transitions. Repeated state names are accepted when
// their transition timestamps differ, allowing multiple tool-call phases.
func (pc *PhaseClock) Note(parser StateTransitioner) {
	if pc == nil || parser == nil {
		return
	}
	t := parser.StateTransitionTime()
	if t.IsZero() {
		return
	}
	pc.NotePhase(parser.StateName(), t)
}

// NotePhase records a transition from structured output, such as native chat.
func (pc *PhaseClock) NotePhase(name string, t time.Time) {
	if pc == nil || t.IsZero() {
		return
	}

	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.startTime.IsZero() || t.Before(pc.startTime) || (!pc.lastTransition.IsZero() && t.Before(pc.lastTransition)) {
		return
	}
	if pc.lastTransition.IsZero() {
		pc.lastTransition = pc.startTime
		pc.activePhase = phaseStateAnswer
	}

	if name == phaseStateThinkingDone {
		if pc.activePhase == phaseStateThinking {
			pc.reasoningDuration += t.Sub(pc.lastTransition)
		} else if pc.reasoningDuration == 0 && pc.thinkStart.IsZero() {
			pc.reasoningDuration += t.Sub(pc.startTime)
		} else {
			pc.closeActivePhase(t)
		}
		pc.thinkStart = t
		pc.activePhase = phaseStateAnswer
		pc.lastTransition = t
		return
	}

	pc.closeActivePhase(t)
	switch name {
	case phaseStateThinking:
		pc.activePhase = phaseStateThinking
		if pc.thinkStart.IsZero() {
			pc.thinkStart = t
		}
	case phaseStateToolCalling:
		pc.activePhase = phaseStateToolCalling
		pc.toolCallStart = t
	default:
		pc.activePhase = phaseStateAnswer
		pc.answerStart = t
	}
	pc.lastTransition = t
}

func (pc *PhaseClock) closeActivePhase(at time.Time) {
	if pc.lastTransition.IsZero() || !at.After(pc.lastTransition) {
		return
	}
	switch pc.activePhase {
	case phaseStateThinking:
		pc.reasoningDuration += at.Sub(pc.lastTransition)
	case phaseStateToolCalling:
		pc.toolCallDuration += at.Sub(pc.lastTransition)
	}
}

// StartTimes returns the first stream time and the most recently observed
// transition time for each phase. Unobserved values are zero.
func (pc *PhaseClock) StartTimes() (startTime, thinkStart, toolCallStart, answerStart time.Time) {
	if pc == nil {
		return
	}
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.startTime, pc.thinkStart, pc.toolCallStart, pc.answerStart
}

// Durations returns accumulated phase durations and assigns remaining
// runner-reported evaluation time to answer or the currently open phase.
// The three returned values always sum to evalDuration.
func (pc *PhaseClock) Durations(evalDuration time.Duration) (answer, reasoning, toolCall time.Duration) {
	answer = evalDuration
	if pc == nil {
		return
	}

	pc.mu.Lock()
	reasoning = pc.reasoningDuration
	toolCall = pc.toolCallDuration
	activePhase := pc.activePhase
	pc.mu.Unlock()

	if evalDuration <= 0 {
		return 0, 0, 0
	}
	if reasoning < 0 {
		reasoning = 0
	}
	if toolCall < 0 {
		toolCall = 0
	}
	if reasoning > evalDuration {
		reasoning = evalDuration
	}
	if toolCall > evalDuration-reasoning {
		toolCall = evalDuration - reasoning
	}
	remaining := evalDuration - reasoning - toolCall
	switch activePhase {
	case phaseStateThinking:
		reasoning += remaining
	case phaseStateToolCalling:
		toolCall += remaining
	}
	answer = evalDuration - reasoning - toolCall
	return
}

// notePhaseClock records the parser's current state transition in the phase
// clock when the parser implements [StateTransitioner]. Parsers without a
// state machine are ignored.
func notePhaseClock(clock *PhaseClock, parser any) {
	if clock == nil {
		return
	}
	st, ok := parser.(StateTransitioner)
	if !ok {
		return
	}
	clock.Note(st)
}

// metricsStateProvider returns the first supplied parser that reports
// per-phase token counts ([StateProvider]), or nil when none do. It lets the
// metrics logging fall back to duration-based splitting when no parser state
// is available.
func metricsStateProvider(parsers ...any) StateProvider {
	for _, p := range parsers {
		if p == nil {
			continue
		}
		// A typed-nil pointer (e.g. *thinkingparser.Parser(nil)) is not equal
		// to nil once stored in an interface, so check for it explicitly.
		if v := reflect.ValueOf(p); v.Kind() == reflect.Ptr && v.IsNil() {
			continue
		}
		if sp, ok := p.(StateProvider); ok {
			return sp
		}
	}
	return nil
}
