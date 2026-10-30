package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// metricsLogFilename is the file name of the token metrics log.
const metricsLogFilename = "tokenmetrics.log"

// MetricsTiming holds the per-request timing breakdown. The split values
// (Answer/Reasoning/ToolCall) sum to the total eval duration and are derived
// from the parser state. The phase start time points (ReasoningStart/
// ToolCallStart/AnswerStart) are the wall-clock times at which the parser
// state first entered the corresponding phase; they are zero when the phase
// was not observed. Values that cannot be determined remain 0.
type MetricsTiming struct {
	TimeToFirstToken time.Duration
	Parse            time.Duration
	Answer           time.Duration
	Reasoning        time.Duration
	ToolCall         time.Duration
	ReasoningStart   time.Time
	ToolCallStart    time.Time
	AnswerStart      time.Time
}

// tokenMetricsLogEntry is one JSONL line written per finished request. The
// duration fields are serialized in milliseconds (see MarshalJSON).
type tokenMetricsLogEntry struct {
	Timestamp          time.Time     `json:"timestamp"`
	Model              string        `json:"model"`
	InputTokens        int           `json:"input_tokens"`
	CachedTokens       *int          `json:"cached_tokens,omitempty"`
	GeneratedTokens    int           `json:"generated_tokens"`
	GeneratedReasoning int           `json:"generated_tokens_reasoning"`
	GeneratedToolCalls int           `json:"generated_tokens_tool_calls"`
	GeneratedAnswer    int           `json:"generated_tokens_answer"`
	TimeToFirstToken   time.Duration `json:"time_to_first_token_ms"`
	ParseDuration      time.Duration `json:"parse_duration_ms"`
	AnswerDuration     time.Duration `json:"answer_generation_duration_ms"`
	ReasoningDuration  time.Duration `json:"reasoning_generation_duration_ms"`
	ToolCallDuration   time.Duration `json:"tool_call_generation_duration_ms"`
	TokensPerSecPrompt float64       `json:"tokens_per_second_parsed"`
	TokensPerSecEval   float64       `json:"tokens_per_second_generated"`
}

// MarshalJSON serializes the entry for the JSONL log, converting the duration
// fields to float64 milliseconds to match the _ms key names.
func (e tokenMetricsLogEntry) MarshalJSON() ([]byte, error) {
	type entryAlias tokenMetricsLogEntry
	return json.Marshal(&struct {
		entryAlias
		TimeToFirstToken  float64 `json:"time_to_first_token_ms"`
		ParseDuration     float64 `json:"parse_duration_ms"`
		AnswerDuration    float64 `json:"answer_generation_duration_ms"`
		ReasoningDuration float64 `json:"reasoning_generation_duration_ms"`
		ToolCallDuration  float64 `json:"tool_call_generation_duration_ms"`
	}{
		entryAlias:        entryAlias(e),
		TimeToFirstToken:  float64(e.TimeToFirstToken / time.Millisecond),
		ParseDuration:     float64(e.ParseDuration / time.Millisecond),
		AnswerDuration:    float64(e.AnswerDuration / time.Millisecond),
		ReasoningDuration: float64(e.ReasoningDuration / time.Millisecond),
		ToolCallDuration:  float64(e.ToolCallDuration / time.Millisecond),
	})
}

// tokenMetricsLogger writes per-request token metrics (JSONL) to a log file.
// It is created in [Server.initMetricsLogging] when
// OLLAMA_LOG_METRICS is enabled; a nil value on [Server] means logging is
// disabled. The write logic ([Server.logTokenMetrics]) appends one JSONL line
// per finished request.
type tokenMetricsLogger struct {
	mu sync.Mutex
	f  *os.File
}

// newTokenMetricsLogger creates the metrics log file at path, creating its
// parent directory when needed, and opens it for appending. The file is
// created with permissions 0644.
func newTokenMetricsLogger(path string) (*tokenMetricsLogger, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &tokenMetricsLogger{f: f}, nil
}

// Close closes the underlying log file.
func (l *tokenMetricsLogger) Close() error {
	if l == nil || l.f == nil {
		return nil
	}
	return l.f.Close()
}

// initMetricsLogging enables token metrics logging when OLLAMA_LOG_METRICS is
// set. It is a no-op when the variable is unset or false. When the log file
// cannot be opened, a warning is logged and metrics logging stays disabled —
// writing token metrics is optional and must never block the server.
func (s *Server) initMetricsLogging() {
	if !envconfig.LogMetrics() {
		return
	}

	path := envconfig.MetricsLogFile()
	if path == "" {
		path = defaultMetricsLogPath()
	}

	l, err := newTokenMetricsLogger(path)
	if err != nil {
		slog.Warn("failed to enable token metrics logging", "path", path, "error", err)
		return
	}

	s.metricsLogger = l
	slog.Info("token metrics logging enabled", "path", path)
}

// defaultMetricsLogPath returns the default token metrics log path, derived
// from the server log path (see app/server/server_unix.go and
// app/server/server_windows.go): on windows it is %LOCALAPPDATA%\Ollama\logs\
// and on darwin/linux $HOME/.ollama/logs\, in either case with the file name
// tokenmetrics.log.
func defaultMetricsLogPath() string {
	if runtime.GOOS == "windows" {
		return metricsLogPath("windows", "", os.Getenv("LOCALAPPDATA"))
	}
	return metricsLogPath(runtime.GOOS, os.Getenv("HOME"), "")
}

// metricsLogPath computes the default token metrics log path for the given
// platform and environment directories. It is split out from
// [defaultMetricsLogPath] so the platform branches can be unit-tested
// independently of the host platform.
func metricsLogPath(goos, home, localAppData string) string {
	if goos == "windows" {
		return filepath.Join(localAppData, "Ollama", "logs", metricsLogFilename)
	}
	return filepath.Join(home, ".ollama", "logs", metricsLogFilename)
}

// tokensPerSecond returns count per second, or 0 when duration is not
// available.
func tokensPerSecond(count int, duration time.Duration) float64 {
	if duration <= 0 {
		return 0
	}
	return float64(count) / duration.Seconds()
}

// logTokenMetrics appends one JSONL line with the token metrics of a finished
// request to the metrics log. Generated-token phase counts are derived from
// tokenized parsed output and reconciled with the runner's authoritative total.
func (s *Server) logTokenMetrics(model string, m api.Metrics, timing MetricsTiming, counts generatedTokenCounts) {
	if s.metricsLogger == nil {
		return
	}

	entry := tokenMetricsLogEntry{
		Timestamp:          time.Now(),
		Model:              model,
		InputTokens:        m.PromptEvalCount,
		CachedTokens:       m.PromptEvalCachedCount,
		GeneratedTokens:    m.EvalCount,
		GeneratedReasoning: counts.reasoning,
		GeneratedToolCalls: counts.toolCalls,
		GeneratedAnswer:    counts.answer,
		TimeToFirstToken:   timing.TimeToFirstToken,
		ParseDuration:      timing.Parse,
		AnswerDuration:     timing.Answer,
		ReasoningDuration:  timing.Reasoning,
		ToolCallDuration:   timing.ToolCall,
		TokensPerSecPrompt: tokensPerSecond(m.PromptEvalCount, m.PromptEvalDuration),
		TokensPerSecEval:   tokensPerSecond(m.EvalCount, m.EvalDuration),
	}

	line, err := json.Marshal(entry)
	if err != nil {
		slog.Warn("failed to marshal token metrics", "model", model, "error", err)
		return
	}

	l := s.metricsLogger
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, err := fmt.Fprintln(l.f, string(line)); err != nil {
		slog.Warn("failed to write token metrics", "model", model, "error", err)
	}
}

func (s *Server) logTokenMetricsWithTokenizer(
	ctx context.Context,
	model string,
	m api.Metrics,
	timing MetricsTiming,
	tokenize func(context.Context, string) ([]int, error),
	text *generatedTokenText,
) {
	if s.metricsLogger == nil {
		return
	}
	if text == nil {
		text = &generatedTokenText{}
	}
	counts, err := countGeneratedTokenPhases(ctx, m.EvalCount, tokenize, *text)
	if err != nil {
		slog.Warn("failed to tokenize generated token phases", "model", model, "error", err)
		return
	}
	s.logTokenMetrics(model, m, timing, counts)
}
