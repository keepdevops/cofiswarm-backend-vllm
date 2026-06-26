// Package vllm implements the InferenceBackend contract against a vLLM
// OpenAI-compatible endpoint (e.g. Docker Model Runner on :12434). It is the
// vLLM-side peer of cofiswarm-backend-mlx and cofiswarm-backend-llama.
//
// Unlike llama-server/mlx, vLLM is a single shared server that REQUIRES a
// `model` field in every request and authenticates with an API key (default
// "EMPTY" in this stack).
package vllm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/keepdevops/cofiswarm-backend-sdk/pkg/backend"
)

// DefaultPort is the shared Docker Model Runner / vLLM OpenAI endpoint port.
const DefaultPort = 12434

// DefaultAPIKey matches config/infer-vllm/vllm.json ("EMPTY"). vLLM ignores it
// unless started with --api-key; sending it is harmless.
const DefaultAPIKey = "EMPTY"

// Backend streams from one vLLM OpenAI-compatible server.
type Backend struct {
	Host         string
	Port         int
	Model        string // REQUIRED by vLLM; sent as the request "model"
	APIKey       string
	AgentID      string
	SystemPrompt string
	MaxTokens    int
	Temperature  float64
	baseURL      string
	client       *http.Client
}

var _ backend.InferenceBackend = (*Backend)(nil)

// NewBackend constructs a vLLM Backend (defaults: 127.0.0.1:12434, api key
// "EMPTY", max_tokens 512, temperature 0.2, 300s stream timeout).
func NewBackend(host string, port int, model, agentID, systemPrompt string, maxTokens int, temperature float64) *Backend {
	if host == "" {
		host = "127.0.0.1"
	}
	if port == 0 {
		port = DefaultPort
	}
	if maxTokens <= 0 {
		maxTokens = backend.DefaultMaxTokens
	}
	if temperature == 0 {
		temperature = 0.2
	}
	return &Backend{
		Host: host, Port: port, Model: model, APIKey: DefaultAPIKey,
		AgentID: agentID, SystemPrompt: systemPrompt,
		MaxTokens: maxTokens, Temperature: temperature,
		baseURL: fmt.Sprintf("http://%s:%d/v1", host, port),
		client:  &http.Client{Timeout: 300 * time.Second},
	}
}

func (b *Backend) messages(prompt string) []map[string]string {
	var msgs []map[string]string
	if b.SystemPrompt != "" {
		msgs = append(msgs, map[string]string{"role": "system", "content": b.SystemPrompt})
	}
	return append(msgs, map[string]string{"role": "user", "content": prompt})
}

func (b *Backend) newRequest(ctx context.Context, path string, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}
	return req, nil
}

// GenerateStream POSTs a streaming chat completion, calling emit for each content
// delta and a final Done chunk. Connection/HTTP failures return a Go error so the
// caller can mark the agent unavailable.
func (b *Backend) GenerateStream(ctx context.Context, req backend.GenerateRequest, emit func(backend.TokenChunk) error) error {
	if b.Model == "" {
		return fmt.Errorf("vllm-backend %s: no model configured (vLLM requires a model)", b.AgentID)
	}
	maxTok := req.MaxTokens
	if maxTok == 0 {
		maxTok = b.MaxTokens
	}
	temp := req.Temperature
	if temp == 0 {
		temp = b.Temperature
	}
	payload := map[string]any{
		"model":       b.Model,
		"messages":    b.messages(req.Prompt),
		"max_tokens":  maxTok,
		"temperature": temp,
		"stream":      true,
	}
	if len(req.Stop) > 0 {
		payload["stop"] = req.Stop
	}
	body, _ := json.Marshal(payload)

	httpReq, err := b.newRequest(ctx, "/chat/completions", body)
	if err != nil {
		return fmt.Errorf("vllm-backend %s: build request: %w", b.AgentID, err)
	}
	resp, err := b.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("vllm-backend %s: connection error: %w", b.AgentID, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("vllm-backend %s: HTTP %d: %s", b.AgentID, resp.StatusCode, truncate(string(raw), 200))
	}

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(line[len("data:"):])
		if data == "[DONE]" {
			break
		}
		var ev struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue // skip a malformed SSE frame
		}
		if len(ev.Choices) > 0 && ev.Choices[0].Delta.Content != "" {
			if err := emit(backend.TokenChunk{Text: ev.Choices[0].Delta.Content}); err != nil {
				return err
			}
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("vllm-backend %s: stream read: %w", b.AgentID, err)
	}
	return emit(backend.TokenChunk{Done: true})
}

// Embed calls /v1/embeddings (vLLM must be serving an embedding model). Returns
// one vector per input text.
func (b *Backend) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	if b.Model == "" {
		return nil, fmt.Errorf("vllm-backend %s: no model configured (vLLM requires a model)", b.AgentID)
	}
	body, _ := json.Marshal(map[string]any{"model": b.Model, "input": texts})
	httpReq, err := b.newRequest(ctx, "/embeddings", body)
	if err != nil {
		return nil, fmt.Errorf("vllm-backend %s: build embed request: %w", b.AgentID, err)
	}
	resp, err := b.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("vllm-backend %s: embed connection error: %w", b.AgentID, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vllm-backend %s: embed HTTP %d: %s", b.AgentID, resp.StatusCode, truncate(string(raw), 200))
	}
	var out struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("vllm-backend %s: embed decode: %w", b.AgentID, err)
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("vllm-backend %s: embed returned %d vectors for %d inputs", b.AgentID, len(out.Data), len(texts))
	}
	vecs := make([][]float32, len(out.Data))
	for i, d := range out.Data {
		vecs[i] = d.Embedding
	}
	return vecs, nil
}

// Health probes GET /health; OK on HTTP 200 (vLLM exposes /health).
func (b *Backend) Health(ctx context.Context) backend.HealthStatus {
	// /health sits at the server root, not under /v1.
	url := strings.TrimSuffix(b.baseURL, "/v1") + "/health"
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}
	cl := &http.Client{Timeout: 5 * time.Second}
	resp, err := cl.Do(req)
	if err != nil {
		return backend.HealthStatus{OK: false, Detail: err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		return backend.HealthStatus{OK: true, Detail: fmt.Sprintf("port %d ok", b.Port)}
	}
	return backend.HealthStatus{OK: false, Detail: fmt.Sprintf("port %d HTTP %d", b.Port, resp.StatusCode)}
}

// Close is a no-op (http.Client needs no teardown).
func (b *Backend) Close() error { return nil }

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}
