package server

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestMetricsLogPath(t *testing.T) {
	cases := []struct {
		name         string
		goos         string
		home         string
		localAppData string
		want         string
	}{
		{
			name: "darwin uses $HOME/.ollama/logs",
			goos: "darwin",
			home: "/Users/test",
			want: filepath.Join("/Users/test", ".ollama", "logs", "tokenmetrics.log"),
		},
		{
			name: "linux uses $HOME/.ollama/logs",
			goos: "linux",
			home: "/home/test",
			want: filepath.Join("/home/test", ".ollama", "logs", "tokenmetrics.log"),
		},
		{
			name:         "windows uses %LOCALAPPDATA%/Ollama/logs",
			goos:         "windows",
			localAppData: `C:\Users\test\AppData\Local`,
			want:         filepath.Join(`C:\Users\test\AppData\Local`, "Ollama", "logs", "tokenmetrics.log"),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := metricsLogPath(tc.goos, tc.home, tc.localAppData)
			if got != tc.want {
				t.Errorf("metricsLogPath(%q, %q, %q) = %q, want %q", tc.goos, tc.home, tc.localAppData, got, tc.want)
			}
			// IsAbs is only meaningful when the case platform matches the
			// host platform (a Windows path is not absolute on unix and vice
			// versa).
			if tc.goos == runtime.GOOS && !filepath.IsAbs(got) {
				t.Errorf("metricsLogPath(%q, %q, %q) = %q, want absolute path", tc.goos, tc.home, tc.localAppData, got)
			}
			if filepath.Base(got) != "tokenmetrics.log" {
				t.Errorf("metricsLogPath(%q, %q, %q) = %q, want file name %q", tc.goos, tc.home, tc.localAppData, got, "tokenmetrics.log")
			}
		})
	}
}

func TestDefaultMetricsLogPath(t *testing.T) {
	got := defaultMetricsLogPath()
	if !filepath.IsAbs(got) {
		t.Errorf("defaultMetricsLogPath() = %q, want absolute path", got)
	}
	if filepath.Base(got) != "tokenmetrics.log" {
		t.Errorf("defaultMetricsLogPath() = %q, want file name %q", got, "tokenmetrics.log")
	}
}

func TestNewTokenMetricsLogger(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "logs", "tokenmetrics.log")

	l, err := newTokenMetricsLogger(path)
	if err != nil {
		t.Fatalf("newTokenMetricsLogger(%q) error: %v", path, err)
	}
	defer l.Close()

	// Write a line through the file handle as a stand-in for the write
	// logic, then verify it is on disk.
	line := `{"model":"test-model"}`
	l.mu.Lock()
	_, err = fmt.Fprintln(l.f, line)
	l.mu.Unlock()
	if err != nil {
		t.Fatalf("write to %q error: %v", path, err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back %q error: %v", path, err)
	}
	if string(data) != line+"\n" {
		t.Errorf("file contents = %q, want %q", string(data), line+"\n")
	}
}

func TestNewTokenMetricsLoggerAppend(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenmetrics.log")

	l, err := newTokenMetricsLogger(path)
	if err != nil {
		t.Fatalf("newTokenMetricsLogger(%q) error: %v", path, err)
	}
	_, err = fmt.Fprintln(l.f, "first")
	if err != nil {
		t.Fatalf("write error: %v", err)
	}
	if err := l.Close(); err != nil {
		t.Fatalf("Close() error: %v", err)
	}

	// Reopening must append, not truncate.
	l2, err := newTokenMetricsLogger(path)
	if err != nil {
		t.Fatalf("second newTokenMetricsLogger(%q) error: %v", path, err)
	}
	defer l2.Close()
	if _, err := fmt.Fprintln(l2.f, "second"); err != nil {
		t.Fatalf("write error: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back %q error: %v", path, err)
	}
	if string(data) != "first\nsecond\n" {
		t.Errorf("file contents = %q, want %q", string(data), "first\nsecond\n")
	}
}

func TestTokenMetricsLoggerCloseNil(t *testing.T) {
	var l *tokenMetricsLogger
	if err := l.Close(); err != nil {
		t.Errorf("(*tokenMetricsLogger)(nil).Close() error: %v", err)
	}
}

func TestLogTokenMetrics(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenmetrics.log")

	l, err := newTokenMetricsLogger(path)
	if err != nil {
		t.Fatalf("newTokenMetricsLogger(%q) error: %v", path, err)
	}
	defer l.Close()

	s := &Server{metricsLogger: l}
	cached := 40
	m := api.Metrics{
		TotalDuration:         2 * time.Second,
		LoadDuration:          500 * time.Millisecond,
		PromptEvalCount:       100,
		PromptEvalCachedCount: &cached,
		PromptEvalDuration:    500 * time.Millisecond,
		EvalCount:             30,
		EvalDuration:          time.Second,
	}
	timing := MetricsTiming{
		TimeToFirstToken: 250 * time.Millisecond,
		Parse:            300 * time.Millisecond,
		Answer:           900 * time.Millisecond,
		Reasoning:        50 * time.Millisecond,
		ToolCall:         50 * time.Millisecond,
	}
	s.logTokenMetrics("test-model", m, timing, generatedTokenCounts{reasoning: 1, toolCalls: 1, answer: 28})

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back %q error: %v", path, err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected exactly 1 line, got %d: %q", len(lines), data)
	}

	var entry map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &entry); err != nil {
		t.Fatalf("line is not valid JSON: %v\nline: %s", err, lines[0])
	}

	cases := map[string]float64{
		"input_tokens":     100,
		"cached_tokens":    40,
		"generated_tokens": 30,
		// 50ms reasoning + 50ms tool call out of 1s total:
		// reasoningTokens = int(30 * 0.05 / 1.0) = 1
		// toolCallTokens  = int(30 * 0.05 / 1.0) = 1
		// answerTokens    = 30 - 1 - 1 = 28
		"generated_tokens_answer":          28,
		"generated_tokens_reasoning":       1,
		"generated_tokens_tool_calls":      1,
		"tokens_per_second_parsed":         200, // 100 / 0.5s
		"tokens_per_second_generated":      30,  // 30 / 1s
		"time_to_first_token_ms":           250,
		"parse_duration_ms":                300,
		"answer_generation_duration_ms":    900,
		"reasoning_generation_duration_ms": 50,
		"tool_call_generation_duration_ms": 50,
	}
	for k, want := range cases {
		got, ok := entry[k].(float64)
		if !ok {
			t.Errorf("missing or non-numeric field %q", k)
			continue
		}
		if got != want {
			t.Errorf("field %q = %v, want %v", k, got, want)
		}
	}
	if got, _ := entry["model"].(string); got != "test-model" {
		t.Errorf("field %q = %v, want %q", "model", entry["model"], "test-model")
	}
}

func TestLogTokenMetricsNoOpWhenDisabled(t *testing.T) {
	s := &Server{metricsLogger: nil}
	s.logTokenMetrics("test-model", api.Metrics{PromptEvalCount: 10, EvalCount: 5}, MetricsTiming{}, generatedTokenCounts{answer: 5})
	// No logger, no file: nothing may be written and no panic may occur.
}

func TestTokensPerSecond(t *testing.T) {
	cases := []struct {
		name     string
		count    int
		duration time.Duration
		want     float64
	}{
		{"prompt eval rate", 100, 500 * time.Millisecond, 200},
		{"eval rate", 30, time.Second, 30},
		{"zero duration yields 0", 100, 0, 0},
		{"negative duration yields 0", 100, -time.Second, 0},
		{"zero count with positive duration yields 0", 0, time.Second, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tokensPerSecond(tc.count, tc.duration); got != tc.want {
				t.Errorf("tokensPerSecond(%d, %v) = %v, want %v", tc.count, tc.duration, got, tc.want)
			}
		})
	}
}

func TestLogTokenMetricsTPandTG(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenmetrics.log")

	l, err := newTokenMetricsLogger(path)
	if err != nil {
		t.Fatalf("newTokenMetricsLogger(%q) error: %v", path, err)
	}
	defer l.Close()

	s := &Server{metricsLogger: l}

	// Metrics with positive durations: tp/tg must be count/duration.
	m := api.Metrics{
		PromptEvalCount:    100,
		PromptEvalDuration: 2 * time.Second,
		EvalCount:          60,
		EvalDuration:       3 * time.Second,
	}
	s.logTokenMetrics("test-model", m, MetricsTiming{}, generatedTokenCounts{answer: 60})

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back %q error: %v", path, err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected exactly 1 line, got %d: %q", len(lines), data)
	}
	var entry map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &entry); err != nil {
		t.Fatalf("line is not valid JSON: %v\nline: %s", err, lines[0])
	}
	if got := entry["tokens_per_second_parsed"].(float64); got != 50 {
		t.Errorf("tokens_per_second_parsed = %v, want 50", got)
	}
	if got := entry["tokens_per_second_generated"].(float64); got != 20 {
		t.Errorf("tokens_per_second_generated = %v, want 20", got)
	}

	// Metrics without durations: tp/tg must fall back to 0.
	s.logTokenMetrics("test-model", api.Metrics{PromptEvalCount: 10, EvalCount: 5}, MetricsTiming{}, generatedTokenCounts{answer: 5})
	data, err = os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back %q error: %v", path, err)
	}
	lines = strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected exactly 2 lines, got %d", len(lines))
	}
	if err := json.Unmarshal([]byte(lines[1]), &entry); err != nil {
		t.Fatalf("line is not valid JSON: %v\nline: %s", err, lines[1])
	}
	if got := entry["tokens_per_second_parsed"].(float64); got != 0 {
		t.Errorf("tokens_per_second_parsed = %v, want 0", got)
	}
	if got := entry["tokens_per_second_generated"].(float64); got != 0 {
		t.Errorf("tokens_per_second_generated = %v, want 0", got)
	}
}

func TestConcurrentWrites(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenmetrics.log")

	l, err := newTokenMetricsLogger(path)
	if err != nil {
		t.Fatalf("newTokenMetricsLogger(%q) error: %v", path, err)
	}
	defer l.Close()

	s := &Server{metricsLogger: l}

	const (
		goroutines = 32
		writes     = 25
	)
	// Every entry is written with the same EvalCount so the total number of
	// generated tokens is deterministic and can be summed for verification.
	const evalCount = 7
	m := api.Metrics{PromptEvalCount: 3, EvalCount: evalCount}

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for range goroutines {
		go func() {
			defer wg.Done()
			for range writes {
				s.logTokenMetrics("test-model", m, MetricsTiming{}, generatedTokenCounts{answer: evalCount})
			}
		}()
	}
	wg.Wait()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back %q error: %v", path, err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	wantLines := goroutines * writes
	if len(lines) != wantLines {
		t.Fatalf("expected exactly %d lines, got %d", wantLines, len(lines))
	}

	totalGenerated := 0
	for i, line := range lines {
		// Each line must be intact, valid JSON written by a single call
		// (no interleaving or truncation from concurrent writes).
		var entry map[string]any
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			t.Fatalf("line %d is not valid JSON: %v\nline: %s", i, err, line)
		}
		if got, _ := entry["model"].(string); got != "test-model" {
			t.Errorf("line %d: model = %v, want %q", i, entry["model"], "test-model")
		}
		got, ok := entry["generated_tokens"].(float64)
		if !ok || int(got) != evalCount {
			t.Errorf("line %d: generated_tokens = %v, want %d", i, entry["generated_tokens"], evalCount)
		}
		totalGenerated += int(got)
	}

	wantTotal := wantLines * evalCount
	if totalGenerated != wantTotal {
		t.Errorf("sum of generated_tokens = %d, want %d", totalGenerated, wantTotal)
	}
}
