package server

import (
	"testing"
	"time"
)

type fakeStateTransitionParser struct {
	state        string
	transitionAt time.Time
}

func (p *fakeStateTransitionParser) StateTransitionTime() time.Time {
	return p.transitionAt
}

func (p *fakeStateTransitionParser) StateName() string {
	return p.state
}

type passthroughLikeParser struct{}

func TestPhaseClockNote(t *testing.T) {
	base := time.Date(2026, 10, 18, 12, 0, 0, 0, time.UTC)

	t.Run("records all phase transition timestamps", func(t *testing.T) {
		clock := &PhaseClock{startTime: base}
		parser := &fakeStateTransitionParser{}
		for _, transition := range []struct {
			state string
			at    time.Time
		}{
			{phaseStateThinking, base.Add(time.Second)},
			{phaseStateThinkingDone, base.Add(4 * time.Second)},
			{phaseStateToolCalling, base.Add(5 * time.Second)},
			{phaseStateAnswer, base.Add(8 * time.Second)},
		} {
			parser.state = transition.state
			parser.transitionAt = transition.at
			clock.Note(parser)
		}

		start, think, tool, answer := clock.StartTimes()
		if !start.Equal(base) || !think.Equal(base.Add(4*time.Second)) ||
			!tool.Equal(base.Add(5*time.Second)) || !answer.Equal(base.Add(8*time.Second)) {
			t.Errorf("StartTimes = (%v, %v, %v, %v), want (%v, %v, %v, %v)",
				start, think, tool, answer, base, base.Add(4*time.Second),
				base.Add(5*time.Second), base.Add(8*time.Second))
		}

		answerDuration, reasoning, toolDuration := clock.Durations(10 * time.Second)
		if reasoning != 3*time.Second || toolDuration != 3*time.Second || answerDuration != 4*time.Second {
			t.Errorf("Durations = (%v, %v, %v), want answer 4s, reasoning 3s, tool 3s",
				answerDuration, reasoning, toolDuration)
		}
	})

	t.Run("repeated tool-call phases are accumulated", func(t *testing.T) {
		clock := &PhaseClock{startTime: base}
		parser := &fakeStateTransitionParser{}
		for _, transition := range []struct {
			state string
			at    time.Time
		}{
			{phaseStateToolCalling, base.Add(time.Second)},
			{phaseStateAnswer, base.Add(3 * time.Second)},
			{phaseStateToolCalling, base.Add(4 * time.Second)},
			{phaseStateAnswer, base.Add(7 * time.Second)},
		} {
			parser.state = transition.state
			parser.transitionAt = transition.at
			clock.Note(parser)
		}

		answer, reasoning, toolCall := clock.Durations(10 * time.Second)
		if reasoning != 0 || toolCall != 5*time.Second || answer != 5*time.Second {
			t.Errorf("Durations = (%v, %v, %v), want answer 5s, reasoning 0, tool 5s",
				answer, reasoning, toolCall)
		}
	})

	t.Run("thinkingDone without an observed thinking transition closes reasoning", func(t *testing.T) {
		clock := &PhaseClock{startTime: base}
		clock.NotePhase(phaseStateThinkingDone, base.Add(2*time.Second))
		answer, reasoning, toolCall := clock.Durations(5 * time.Second)
		if reasoning != 2*time.Second || toolCall != 0 || answer != 3*time.Second {
			t.Errorf("Durations = (%v, %v, %v), want answer 3s, reasoning 2s, tool 0",
				answer, reasoning, toolCall)
		}
	})

	t.Run("zero transition and unknown state do not split durations", func(t *testing.T) {
		clock := &PhaseClock{startTime: base}
		clock.Note(&fakeStateTransitionParser{state: phaseStateThinking})
		clock.Note(&fakeStateTransitionParser{state: "unknown", transitionAt: base.Add(time.Second)})
		answer, reasoning, toolCall := clock.Durations(time.Second)
		if answer != time.Second || reasoning != 0 || toolCall != 0 {
			t.Errorf("Durations = (%v, %v, %v), want answer 1s, reasoning 0, tool 0",
				answer, reasoning, toolCall)
		}
	})

	t.Run("nil clock and parser are no-ops", func(t *testing.T) {
		var clock *PhaseClock
		clock.Note(&fakeStateTransitionParser{state: phaseStateThinking, transitionAt: base})
		clock.NotePhase(phaseStateThinking, base)
		clock.Begin()
		notePhaseClock(clock, &fakeStateTransitionParser{state: phaseStateThinking, transitionAt: base})
		notePhaseClock(&PhaseClock{}, nil)
	})

	t.Run("parser without a state machine is ignored", func(t *testing.T) {
		clock := &PhaseClock{startTime: base}
		notePhaseClock(clock, passthroughLikeParser{})
		if _, think, tool, answer := clock.StartTimes(); !think.IsZero() || !tool.IsZero() || !answer.IsZero() {
			t.Errorf("phase starts = (%v, %v, %v), want zero", think, tool, answer)
		}
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

	t.Run("answer, reasoning, and tool durations are all represented", func(t *testing.T) {
		clock := &PhaseClock{startTime: base}
		clock.NotePhase(phaseStateThinking, base)
		clock.NotePhase(phaseStateThinkingDone, base.Add(3*time.Second))
		clock.NotePhase(phaseStateToolCalling, base.Add(4*time.Second))
		clock.NotePhase(phaseStateAnswer, base.Add(6*time.Second))

		answer, reasoning, toolCall := clock.Durations(10 * time.Second)
		if reasoning != 3*time.Second || toolCall != 2*time.Second || answer != 5*time.Second {
			t.Errorf("Durations = (%v, %v, %v), want answer 5s, reasoning 3s, tool 2s",
				answer, reasoning, toolCall)
		}
	})

	t.Run("tool call without thinking uses only its observed interval", func(t *testing.T) {
		clock := &PhaseClock{startTime: base}
		clock.NotePhase(phaseStateToolCalling, base.Add(2*time.Second))
		clock.NotePhase(phaseStateAnswer, base.Add(4*time.Second))
		answer, reasoning, toolCall := clock.Durations(5 * time.Second)
		if reasoning != 0 || toolCall != 2*time.Second || answer != 3*time.Second {
			t.Errorf("Durations = (%v, %v, %v), want answer 3s, reasoning 0, tool 2s",
				answer, reasoning, toolCall)
		}
	})

	t.Run("phase durations are clamped to runner duration", func(t *testing.T) {
		clock := &PhaseClock{startTime: base}
		clock.NotePhase(phaseStateThinking, base)
		clock.NotePhase(phaseStateThinkingDone, base.Add(8*time.Second))
		clock.NotePhase(phaseStateToolCalling, base.Add(8*time.Second))
		clock.NotePhase(phaseStateAnswer, base.Add(9*time.Second))
		answer, reasoning, toolCall := clock.Durations(time.Second)
		if reasoning != time.Second || toolCall != 0 || answer != 0 {
			t.Errorf("Durations = (%v, %v, %v), want (0, 1s, 0)", answer, reasoning, toolCall)
		}
	})

	t.Run("no phase transitions attributes all duration to answer", func(t *testing.T) {
		clock := &PhaseClock{startTime: base}
		answer, reasoning, toolCall := clock.Durations(time.Second)
		if answer != time.Second || reasoning != 0 || toolCall != 0 {
			t.Errorf("Durations = (%v, %v, %v), want answer 1s, reasoning 0, tool 0",
				answer, reasoning, toolCall)
		}
	})

	t.Run("nil clock attributes all duration to answer", func(t *testing.T) {
		var clock *PhaseClock
		answer, reasoning, toolCall := clock.Durations(time.Second)
		if answer != time.Second || reasoning != 0 || toolCall != 0 {
			t.Errorf("Durations = (%v, %v, %v), want answer 1s, reasoning 0, tool 0",
				answer, reasoning, toolCall)
		}
	})
}
