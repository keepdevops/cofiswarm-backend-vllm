package vllm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/keepdevops/cofiswarm-backend-sdk/pkg/backend"
)

func newBackendOn(t *testing.T, model string, h http.HandlerFunc) (*Backend, func()) {
	t.Helper()
	srv := httptest.NewServer(h)
	b := NewBackend("", 0, model, "scout", "You are Scout.", 64, 0)
	b.baseURL = srv.URL + "/v1"
	return b, srv.Close
}

func TestBackendGenerateStream(t *testing.T) {
	var sentModel, auth string
	var roles []string
	b, stop := newBackendOn(t, "Qwen2.5-7B", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			w.WriteHeader(404)
			return
		}
		auth = r.Header.Get("Authorization")
		var body struct {
			Model    string `json:"model"`
			Messages []struct {
				Role string `json:"role"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		sentModel = body.Model
		for _, m := range body.Messages {
			roles = append(roles, m.Role)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, tok := range []string{"SH", "IP"} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", tok)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	defer stop()

	var out string
	var sawDone bool
	err := b.GenerateStream(context.Background(),
		backend.GenerateRequest{Prompt: "rate it"},
		func(c backend.TokenChunk) error {
			out += c.Text
			if c.Done {
				sawDone = true
			}
			return nil
		})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if out != "SHIP" || !sawDone {
		t.Errorf("out=%q sawDone=%v, want SHIP + Done", out, sawDone)
	}
	if sentModel != "Qwen2.5-7B" {
		t.Errorf("model sent=%q, want Qwen2.5-7B (vLLM requires it)", sentModel)
	}
	if auth != "Bearer EMPTY" {
		t.Errorf("auth header=%q, want Bearer EMPTY", auth)
	}
	if len(roles) != 2 || roles[0] != "system" || roles[1] != "user" {
		t.Errorf("roles=%v, want [system user]", roles)
	}
}

func TestBackendNoModelFailsLoud(t *testing.T) {
	b := NewBackend("127.0.0.1", 0, "", "scout", "", 0, 0)
	err := b.GenerateStream(context.Background(), backend.GenerateRequest{Prompt: "p"},
		func(backend.TokenChunk) error { return nil })
	if err == nil {
		t.Fatal("want error when no model is configured")
	}
	if _, eerr := b.Embed(context.Background(), []string{"a"}); eerr == nil {
		t.Error("Embed should also require a model")
	}
}

func TestBackendGenerateStreamEmitAbort(t *testing.T) {
	b, stop := newBackendOn(t, "m", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 5; i++ {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	defer stop()

	want := fmt.Errorf("client gone")
	got := 0
	err := b.GenerateStream(context.Background(), backend.GenerateRequest{Prompt: "p"},
		func(backend.TokenChunk) error { got++; return want })
	if err != want {
		t.Errorf("err=%v, want %v", err, want)
	}
	if got != 1 {
		t.Errorf("emit called %d times, want 1 (abort on first error)", got)
	}
}

func TestBackendEmbed(t *testing.T) {
	b, stop := newBackendOn(t, "embed-model", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			w.WriteHeader(404)
			return
		}
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2]},{"embedding":[0.3,0.4]}]}`))
	})
	defer stop()

	vecs, err := b.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(vecs) != 2 || vecs[1][0] != 0.3 {
		t.Errorf("vecs=%v", vecs)
	}
}

func TestBackendHealth(t *testing.T) {
	b, stop := newBackendOn(t, "m", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" { // root, not under /v1
			w.WriteHeader(200)
			return
		}
		w.WriteHeader(404)
	})
	defer stop()

	if h := b.Health(context.Background()); !h.OK {
		t.Errorf("health not OK: %+v", h)
	}

	bad := NewBackend("127.0.0.1", 1, "m", "scout", "", 0, 0) // nothing listening
	if h := bad.Health(context.Background()); h.OK {
		t.Error("health should be false when unreachable")
	}
	if err := bad.Close(); err != nil {
		t.Errorf("close: %v", err)
	}
}
