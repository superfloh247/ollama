package server

import (
	"reflect"
	"sync"
	"time"
)

// The normalized phase names a [StateTransitioner] reports via StateName.
// [PhaseClock.Note] attributes each state transition time point to the phase
// bucket the name belongs to: "thinking"/"thinkingDone" to thinkStart,
// "toolCalling" to toolCallStart, and everything else to answerStart.
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

// PhaseClock collects the wall-clock time points of the parser phase
// transitions observed during a single request. It is created per request in
// the completion handlers and fed from the parser closure on every stream
// chunk ([Note]); the time points are available at the Done signal via
// [StartTimes]. The clock is safe for concurrent use.
type PhaseClock struct {
	mu            sync.Mutex
	startTime     time.Time
	thinkStart    time.Time
	toolCallStart time.Time
	answerStart   time.Time
	seen          map[string]bool
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

// Note attributes the parser's last state transition time point to the phase
// bucket the parser's current state belongs to. The time point of the most
// recent transition of a bucket wins, so when both "thinking" and
// "thinkingDone" are observed the recorded thinkStart is the transition to
// "thinkingDone" (the end of the reasoning phase). A state name that was
// already recorded is not processed again. A zero transition time (no
// transition yet) is ignored.
func (pc *PhaseClock) Note(parser StateTransitioner) {
	if pc == nil || parser == nil {
		return
	}
	t := parser.StateTransitionTime()
	if t.IsZero() {
		return
	}
	name := parser.StateName()

	pc.mu.Lock()
	defer pc.mu.Unlock()
	if pc.seen == nil {
		pc.seen = make(map[string]bool)
	}
	if pc.seen[name] {
		return
	}
	pc.seen[name] = true
	switch name {
	case phaseStateThinking, phaseStateThinkingDone:
		pc.thinkStart = t
	case phaseStateToolCalling:
		pc.toolCallStart = t
	default:
		pc.answerStart = t
	}
}

// StartTimes returns the recorded time points: the time of the first stream
// chunk and the phase transition time points. Values that were not observed
// are zero; a nil clock returns all zeros.
func (pc *PhaseClock) StartTimes() (startTime, thinkStart, toolCallStart, answerStart time.Time) {
	if pc == nil {
		return
	}
	pc.mu.Lock()
	defer pc.mu.Unlock()
	return pc.startTime, pc.thinkStart, pc.toolCallStart, pc.answerStart
}

// Durations derives the phase durations from the recorded time points and
// the runner-reported total generation duration (evalDuration):
//
//	reasoning = thinkStart - startTime (0 if not observed)
//	toolCall  = toolCallStart - thinkStart (from startTime if no thinking
//	            phase was observed; 0 if not observed)
//	answer    = evalDuration - reasoning - toolCall
//
// If reasoning and toolCall together exceed evalDuration, they are clamped
// so that the answer stays non-negative; the three returned values always
// sum to evalDuration. A nil clock returns (evalDuration, 0, 0).
func (pc *PhaseClock) Durations(evalDuration time.Duration) (answer, reasoning, toolCall time.Duration) {
	answer = evalDuration
	if pc == nil {
		return
	}

	pc.mu.Lock()
	startTime, thinkStart, toolCallStart := pc.startTime, pc.thinkStart, pc.toolCallStart
	pc.mu.Unlock()

	if !startTime.IsZero() && !thinkStart.IsZero() {
		if d := thinkStart.Sub(startTime); d > 0 {
			reasoning = d
		}
	}
	if !toolCallStart.IsZero() {
		base := thinkStart
		if base.IsZero() {
			base = startTime
		}
		if !base.IsZero() {
			if d := toolCallStart.Sub(base); d > 0 {
				toolCall = d
			}
		}
	}

	if reasoning > evalDuration {
		reasoning = evalDuration
	}
	if reasoning+toolCall > evalDuration {
		toolCall = evalDuration - reasoning
	}
	answer = evalDuration - reasoning - toolCall
	if answer < 0 {
		answer = 0
	}
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
