"""Local LLM client. Default provider: Ollama (quantized SLMs, fully on-device).

Swap the provider in config.yaml. A MockLLM lets you test the pipeline
end-to-end without any model installed.
"""
from __future__ import annotations


class OllamaLLM:
    def __init__(self, model: str = "llama3.2", temperature: float = 0.1,
                 max_tokens: int = 900):
        import ollama
        self.client = ollama.Client()
        self.model = model
        self.options = {"temperature": temperature, "num_predict": max_tokens}

    def generate(self, system: str, prompt: str) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            options=self.options,
        )
        return resp["message"]["content"]


class MockLLM:
    """Deterministic stand-in so the pipeline runs without a model server."""
    model = "mock"

    def generate(self, system: str, prompt: str) -> str:
        return (
            '{"scores": {"academic_readiness": 3, "research_potential": 3, '
            '"motivation_fit": 3, "recommendations": 3, "experience_skills": 3}, '
            '"recommendation": "borderline", '
            '"justification": "MockLLM output — install Ollama and set llm.provider: ollama.", '
            '"evidence_cited": []}'
        )


def get_llm(cfg: dict):
    provider = cfg.get("provider", "ollama")
    if provider == "mock":
        return MockLLM()
    try:
        return OllamaLLM(model=cfg.get("model", "llama3.2"),
                         temperature=cfg.get("temperature", 0.1),
                         max_tokens=cfg.get("max_tokens", 900))
    except Exception as e:  # ollama not installed / server down
        print(f"[warn] Ollama unavailable ({e}); falling back to MockLLM")
        return MockLLM()
