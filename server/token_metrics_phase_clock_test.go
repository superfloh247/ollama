package server

import (
	"testing"
	"time"
)

// fakeStateTransitionParser is a [StateTransitioner] with simulated
// transition time points: the test advances it state by state without an
// actual clock, mirroring what the parser state machine reports during a
// request.
type fakeStateTransitionParser struct {
	state         string
	transitionAt  time.Time
	transitionSeq int
}

// SetState simulates a state transition of the parser: it records a new
// transition time point (fixed base + sequence offset, so consecutive
// transitions are strictly increasing and fully deterministic) and switches
// the state.
func (p *fakeStateTransitionParser) SetState(state string, base time.Time) {
	p.transitionSeq++
	p.transitionAt = base.Add(time.Duration(p.transitionSeq) * time.Millisecond)
	p.state = state
}

// SetStateAt simulates a state transition to state at an explicit
// deterministic time point.
func (p *fakeStateTransitionParser) SetStateAt(state string, at time.Time) {
	p.state = state
	p.transitionAt = at
}

func (p *fakeStateTransitionParser) StateTransitionTime() time.Time {
	return p.transitionAt
}

func (p *fakeStateTransitionParser) StateName() string {
	return p.state
}

// passthroughLikeParser has no state machine: it does not implement
// [StateTransitioner].
type passthroughLikeParser struct{}

func TestPhaseClockNote(t *testing.T) {
	base := time.Date(2026, 10, 18, 12, 0, 0, 0, time.UTC)

	t.Run("timestamps are attributed to the matching phase", func(t *testing.T) {
		clock := &PhaseClock{}
		parser := &fakeStateTransitionParser{state: "answer"}
		parser.SetState("thinking", base)
		clock.Note(parser)
		parser.SetState("thinkingDone", base)
		clock.Note(parser)
		parser.SetState("toolCalling", base)
		clock.Note(parser)
		parser.SetState("answer", base)
		clock.Note(parser)

		start, think, tool, answer := clock.StartTimes()
		// base+2ms: the "thinkingDone" transition (the later of the two
		// transitions mapped to the think bucket).
		if want := base.Add(2 * time.Millisecond); !think.Equal(want) {
			t.Errorf("thinkStart = %v, want %v (the thinkingDone transition)", think, want)
		}
		if want := base.Add(3 * time.Millisecond); !tool.Equal(want) {
			t.Errorf("toolCallStart = %v, want %v", tool, want)
		}
		if want := base.Add(4 * time.Millisecond); !answer.Equal(want) {
			t.Errorf("answerStart = %v, want %v", answer, want)
		}
		if !start.IsZero() {
			t.Errorf("startTime = %v, want zero (Begin was not called)", start)
		}
	})

	t.Run("the most recent transition of a bucket wins", func(t *testing.T) {
		// Both "thinking" and "thinkingDone" map to the think bucket; the
		// most recent transition wins, so the recorded thinkStart is the
		// thinkingDone transition (the end of the reasoning phase).
		clock := &PhaseClock{}
		parser := &fakeStateTransitionParser{state: "answer"}
		parser.SetState("thinking", base)
		clock.Note(parser)
		parser.SetState("thinkingDone", base)
		clock.Note(parser)

		_, think, _, _ := clock.StartTimes()
		if want := base.Add(2 * time.Millisecond); !think.Equal(want) {
			t.Errorf("thinkStart = %v, want %v", think, want)
		}
	})

	t.Run("repeated state names are not recorded again", func(t *testing.T) {
		clock := &PhaseClock{}
		parser := &fakeStateTransitionParser{state: "answer"}
		parser.SetState("thinking", base)
		clock.Note(parser)
		_, want, _, _ := clock.StartTimes()

		// The parser stayed in "thinking" across more chunks (the fake
		// reports a newer time point); the recorded time must not change.
		parser.SetState("thinking", base)
		clock.Note(parser)

		_, think, _, _ := clock.StartTimes()
		if !think.Equal(want) {
			t.Errorf("thinkStart = %v, want unchanged %v", think, want)
		}
	})

	t.Run("zero transition time is ignored", func(t *testing.T) {
		clock := &PhaseClock{}
		parser := &fakeStateTransitionParser{state: "thinking"} // no transition yet
		clock.Note(parser)

		start, think, tool, answer := clock.StartTimes()
		if !start.IsZero() || !think.IsZero() || !tool.IsZero() || !answer.IsZero() {
			t.Errorf("StartTimes = (%v, %v, %v, %v), want all zero", start, think, tool, answer)
		}
	})

	t.Run("nil clock and nil parser are no-ops", func(t *testing.T) {
		parser := &fakeStateTransitionParser{state: "thinking", transitionAt: base}
		var clock *PhaseClock
		clock.Note(parser)
		clock.Begin()
		notePhaseClock(clock, parser)
		notePhaseClock(&PhaseClock{}, nil)
	})
}

func TestPhaseClockBegin(t *testing.T) {
	t.Run("the first call records the current time and later calls keep it", func(t *testing.T) {
		clock := &PhaseClock{}
		before := time.Now()
		clock.Begin()
		clock.Begin()
		start, _, _, _ := clock.StartTimes()
		if start.Before(before) || start.After(time.Now()) {
			t.Errorf("startTime = %v, want within [%v, now]", start, before)
		}
	})

	t.Run("a nil clock is a no-op", func(t *testing.T) {
		var clock *PhaseClock
		clock.Begin()
		start, _, _, _ := clock.StartTimes()
		if !start.IsZero() {
			t.Errorf("startTime = %v, want zero", start)
		}
	})
}

func TestPhaseClockDurations(t *testing.T) {
	base := time.Date(2026, 10, 18, 12, 0, 0, 0, time.UTC)
	evalDuration := time.Second

	// newClock builds a clock with fully deterministic, explicit time points
	// (no real clock involved): startTime is fixed, the think/toolCall
	// buckets are filled with fake transition time points.
	newClock := func(thinkStart, toolCallStart time.Time) *PhaseClock {
		return &PhaseClock{startTime: base, thinkStart: thinkStart, toolCallStart: toolCallStart}
	}

	t.Run("thinking only: reasoning = thinkStart - startTime", func(t *testing.T) {
		// Plan scenario (a): start 12:00:00, thinkStart 12:00:05, eval 10s
		// → reasoning 5s, toolCall 0, answer 5s.
		clock := newClock(base.Add(5*time.Second), time.Time{})
		answer, reasoning, toolCall := clock.Durations(10 * time.Second)
		if reasoning != 5*time.Second || toolCall != 0 || answer != 5*time.Second {
			t.Errorf("Durations = (answer %v, reasoning %v, toolCall %v), want (5s, 5s, 0)", answer, reasoning, toolCall)
		}
		if sum := reasoning + toolCall + answer; sum != 10*time.Second {
			t.Errorf("sum = %v, want 10s", sum)
		}
	})

	t.Run("thinking plus tool calling", func(t *testing.T) {
		// Plan scenario (b): start 12:00:00, thinkStart 12:00:05,
		// toolCallStart 12:00:10, eval 12s → reasoning 5s, toolCall 5s,
		// answer 2s.
		clock := newClock(base.Add(5*time.Second), base.Add(10*time.Second))
		answer, reasoning, toolCall := clock.Durations(12 * time.Second)
		if reasoning != 5*time.Second || toolCall != 5*time.Second || answer != 2*time.Second {
			t.Errorf("Durations = (answer %v, reasoning %v, toolCall %v), want (2s, 5s, 5s)", answer, reasoning, toolCall)
		}
		if sum := reasoning + toolCall + answer; sum != 12*time.Second {
			t.Errorf("sum = %v, want 12s", sum)
		}
	})

	t.Run("no thinking, no tool calling: everything is the answer", func(t *testing.T) {
		// Plan scenario (c): no transitions recorded at all.
		clock := newClock(time.Time{}, time.Time{})
		answer, reasoning, toolCall := clock.Durations(evalDuration)
		if reasoning != 0 || toolCall != 0 || answer != evalDuration {
			t.Errorf("Durations = (answer %v, reasoning %v, toolCall %v), want (1s, 0, 0)", answer, reasoning, toolCall)
		}
	})

	t.Run("tool calling without a thinking phase is measured from the start", func(t *testing.T) {
		// Plan scenario (d): start 12:00:00, toolCallStart 12:00:02, eval 5s
		// → reasoning 0, toolCall 2s, answer 3s.
		clock := newClock(time.Time{}, base.Add(2*time.Second))
		answer, reasoning, toolCall := clock.Durations(5 * time.Second)
		if reasoning != 0 || toolCall != 2*time.Second || answer != 3*time.Second {
			t.Errorf("Durations = (answer %v, reasoning %v, toolCall %v), want (3s, 0, 2s)", answer, reasoning, toolCall)
		}
	})

	t.Run("durations are clamped so the parts sum to the eval duration", func(t *testing.T) {
		// The clock can report time points that exceed the runner-reported
		// eval duration (for example because the stream chunks arrive after
		// the runner stopped measuring); the parts must be clamped to stay
		// non-negative and to sum to the eval duration.
		clock := newClock(base.Add(8*time.Second), base.Add(9*time.Second))
		answer, reasoning, toolCall := clock.Durations(evalDuration)
		if reasoning != evalDuration || toolCall != 0 || answer != 0 {
			t.Errorf("Durations = (answer %v, reasoning %v, toolCall %v), want (0, 1s, 0)", answer, reasoning, toolCall)
		}
		if sum := reasoning + toolCall + answer; sum != evalDuration {
			t.Errorf("sum = %v, want %v", sum, evalDuration)
		}
	})

	t.Run("a tool-call phase exceeding the remainder is clamped", func(t *testing.T) {
		// reasoning 3s + toolCall 5s on a 6s eval: toolCall is clamped to the
		// remaining 3s, answer 0.
		clock := newClock(base.Add(3*time.Second), base.Add(8*time.Second))
		answer, reasoning, toolCall := clock.Durations(6 * time.Second)
		if reasoning != 3*time.Second || toolCall != 3*time.Second || answer != 0 {
			t.Errorf("Durations = (answer %v, reasoning %v, toolCall %v), want (0, 3s, 3s)", answer, reasoning, toolCall)
		}
		if sum := reasoning + toolCall + answer; sum != 6*time.Second {
			t.Errorf("sum = %v, want 6s", sum)
		}
	})

	t.Run("a reasoning phase alone larger than the eval duration is clamped", func(t *testing.T) {
		clock := newClock(base.Add(4*time.Second), time.Time{})
		answer, reasoning, toolCall := clock.Durations(2 * time.Second)
		if reasoning != 2*time.Second || toolCall != 0 || answer != 0 {
			t.Errorf("Durations = (answer %v, reasoning %v, toolCall %v), want (0, 2s, 0)", answer, reasoning, toolCall)
		}
	})

	t.Run("a nil clock attributes everything to the answer", func(t *testing.T) {
		var clock *PhaseClock
		answer, reasoning, toolCall := clock.Durations(evalDuration)
		if reasoning != 0 || toolCall != 0 || answer != evalDuration {
			t.Errorf("Durations = (answer %v, reasoning %v, toolCall %v), want (1s, 0, 0)", answer, reasoning, toolCall)
		}
	})
}

func TestNotePhaseClock(t *testing.T) {
	t.Run("parsers without a state machine are ignored", func(t *testing.T) {
		clock := &PhaseClock{}
		notePhaseClock(clock, passthroughLikeParser{})

		start, think, tool, answer := clock.StartTimes()
		if !start.IsZero() || !think.IsZero() || !tool.IsZero() || !answer.IsZero() {
			t.Errorf("StartTimes = (%v, %v, %v, %v), want all zero", start, think, tool, answer)
		}
	})

	t.Run("implementors are recorded", func(t *testing.T) {
		base := time.Date(2026, 10, 18, 12, 0, 0, 0, time.UTC)
		clock := &PhaseClock{}
		parser := &fakeStateTransitionParser{state: "toolCalling", transitionAt: base}
		notePhaseClock(clock, parser)

		_, _, tool, _ := clock.StartTimes()
		if !tool.Equal(base) {
			t.Errorf("toolCallStart = %v, want %v", tool, base)
		}
	})
}
