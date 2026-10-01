import asyncio
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .causal_analyzer import CausalChainAnalyzer
from .knn_router import KNNRouter
from .metrics import CausalMetrics
from .models import (
    AnalysisRequest,
    AnalysisResponse,
    RootCauseResponse,
    SymptomRequest,
)
from .models_layer import ModelLayer, OpenRouterError

app = FastAPI(
    title="Log Anomaly Detection System",
    description=(
        "An AI-assisted incident triage system that infers likely root causes from symptom-only logs, "
        "scores confidence, and explains the reasoning behind each diagnosis."
    ),
    version="2.0.0",
    openapi_tags=[
        {"name": "diagnostics", "description": "Incident analysis and root-cause inference endpoints."},
        {"name": "portfolio", "description": "Demo and evaluation endpoints for showcasing the solution."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = KNNRouter()
analyzer = CausalChainAnalyzer()
model_layer = ModelLayer()

HTML_PAGE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Symptom → Root Cause Detector</title>
    <style>
      :root {
        --bg: #0f172a;
        --panel: #111827;
        --panel-alt: #1f2937;
        --card: #f8fafc;
        --primary: #2563eb;
        --primary-soft: #dbeafe;
        --text: #0f172a;
        --muted: #475569;
        --success: #047857;
        --border: #dfe7f1;
      }
      body {
        font-family: Arial, sans-serif;
        margin: 0;
        background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
        color: var(--text);
      }
      .container {
        max-width: 860px;
        margin: 40px auto;
        padding: 24px;
      }
      .hero {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        padding: 28px;
      }
      h1 { margin-top: 0; }
      .subtitle { color: var(--muted); font-size: 1.02rem; }
      textarea {
        width: 100%;
        min-height: 140px;
        margin-top: 16px;
        padding: 14px 16px;
        border: 1px solid var(--border);
        border-radius: 12px;
        font-size: 1rem;
        resize: vertical;
        background: #fbfdff;
      }
      button {
        margin-top: 14px;
        border: none;
        background: var(--primary);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        cursor: pointer;
      }
      .result {
        margin-top: 22px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px;
      }
      .badge {
        display: inline-block;
        padding: 6px 12px;
        background: var(--primary-soft);
        color: var(--primary);
        border-radius: 999px;
        font-weight: 700;
      }
      .meta { color: var(--muted); }
      ul { line-height: 1.7; }
      code {
        background: #eef2ff;
        padding: 2px 6px;
        border-radius: 6px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="hero">
        <h1>Symptom → Root Cause Detector</h1>
        <p class="subtitle">
          Paste an observed symptom or effect from production logs. Automatic routing uses measured backend quality and cost; a free model can also be selected explicitly.
        </p>
        <textarea id="logText" placeholder="Example: Database connection timeout while querying users"></textarea>
        <label for="llmModel">Free OpenRouter model</label>
        <select id="llmModel" disabled>
          <option value="">Automatic quality/cost routing</option>
        </select>
        <button onclick="submitLogs()">Analyze symptom</button>
        <div id="result" class="result" style="display:none"></div>
      </div>
    </div>

    <script>
      async function loadModels() {
        const select = document.getElementById('llmModel');
        try {
          const response = await fetch('/openrouter/models');
          const data = await response.json();
          if (!data.configured) return;

          select.disabled = false;
          for (const model of data.models) {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = model.name;
            if (model.id === data.default_model) option.selected = true;
            select.appendChild(option);
          }
        } catch (error) {
          console.error('Unable to load OpenRouter models', error);
        }
      }

      async function submitLogs() {
        const text = document.getElementById('logText').value.trim();
        if (!text) {
          alert('Please enter a symptom log first.');
          return;
        }

        const selectedModel = document.getElementById('llmModel').value;
        const log = {
          timestamp: new Date().toISOString(),
          source: 'user',
          level: 'ERROR',
          message: text
        };
        const response = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            logs: [log],
            llm_model: selectedModel || null
          })
        });

        const data = await response.json();
        const output = document.getElementById('result');
        output.style.display = 'block';
        output.textContent = `Model: ${data.model_used}\n\n${data.explanation}`;
        if (!response.ok) output.textContent = data.detail || 'Analysis failed.';
      }

      loadModels();
    </script>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["portfolio"])
async def root():
    return HTML_PAGE


@app.get("/health", tags=["diagnostics"])
async def health():
    return {"status": "healthy"}


@app.get("/openrouter/models", tags=["diagnostics"])
async def openrouter_models():
    """List currently available OpenRouter models with free pricing."""
    if not model_layer.openrouter_configured:
        return {"configured": False, "default_model": None, "models": []}
    try:
        models = await asyncio.to_thread(model_layer.list_free_openrouter_models)
        return {
            "configured": True,
            "default_model": os.getenv("OPENROUTER_MODEL"),
            "models": models,
        }
    except OpenRouterError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/analyze", response_model=AnalysisResponse, tags=["diagnostics"])
async def analyze_logs(request: AnalysisRequest):
    try:
        logs_dict = [log.model_dump() for log in request.logs]
        causal_chain = analyzer.analyze(logs_dict)
        if request.llm_model:
            selected_model = request.llm_model
            if not selected_model.endswith(":free"):
                raise HTTPException(
                    status_code=400,
                    detail="Only OpenRouter model IDs ending in ':free' are allowed for manual overrides.",
                )
            if not model_layer.openrouter_configured:
                raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not configured.")
            try:
                diagnosis = await asyncio.to_thread(
                    model_layer.openrouter_diagnosis,
                    selected_model,
                    logs_dict,
                    causal_chain,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except OpenRouterError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            model_name = selected_model
            root_cause_category = diagnosis["category"]
            explanation = diagnosis["explanation"]
            routing = {
                "backend": f"openrouter:{selected_model}",
                "actual_backend": f"openrouter:{selected_model}",
                "policy_source": "manual_override",
                "selection_reason": "explicit_request",
                "estimated_quality": None,
                "estimated_cost_usd": None,
                "estimated_latency_ms": None,
                "actual_cost_usd": diagnosis["cost_usd"],
                "actual_latency_ms": round(diagnosis["latency_ms"], 2),
                "quality_target": router.min_quality,
                "max_cost_usd": router.max_cost_usd,
            }
        else:
            candidate_models = [
                model_id.strip()
                for model_id in os.getenv(
                    "ROUTER_CANDIDATE_MODELS", os.getenv("OPENROUTER_MODEL", "")
                ).split(",")
                if model_id.strip()
            ]
            available_backends = ["rule_based"]
            if model_layer.openrouter_configured:
                available_backends.extend(
                    f"openrouter:{model_id}" for model_id in candidate_models
                )
            routing = router.predict(
                logs_dict,
                request.context or "",
                available_backends=available_backends,
            )
            backend = routing["backend"]
            if backend.startswith("openrouter:"):
                selected_model = backend.removeprefix("openrouter:")
                started = time.perf_counter()
                try:
                    diagnosis = await asyncio.to_thread(
                        model_layer.openrouter_diagnosis,
                        selected_model,
                        logs_dict,
                        causal_chain,
                    )
                except OpenRouterError as exc:
                    model_name = "rule_based"
                    root_cause = causal_chain.get("root_cause") or {}
                    root_cause_category = root_cause.get("category", "unknown")
                    explanation = model_layer.rule_based_analysis(logs_dict, causal_chain)
                    routing["actual_backend"] = "rule_based"
                    routing["actual_cost_usd"] = None
                    routing["actual_latency_ms"] = round(
                        (time.perf_counter() - started) * 1000.0, 2
                    )
                    routing["execution_fallback"] = "provider_request_failed"
                    routing["provider_error"] = str(exc)
                else:
                    model_name = selected_model
                    root_cause_category = diagnosis["category"]
                    explanation = diagnosis["explanation"]
                    routing["actual_backend"] = backend
                    routing["actual_cost_usd"] = diagnosis["cost_usd"]
                    routing["actual_latency_ms"] = round(diagnosis["latency_ms"], 2)
            else:
                model_name = "rule_based"
                root_cause = causal_chain.get("root_cause") or {}
                root_cause_category = root_cause.get("category", "unknown")
                started = time.perf_counter()
                explanation = model_layer.rule_based_analysis(logs_dict, causal_chain)
                routing["actual_backend"] = "rule_based"
                routing["actual_cost_usd"] = 0.0
                routing["actual_latency_ms"] = round(
                    (time.perf_counter() - started) * 1000.0, 2
                )

        return AnalysisResponse(
            model_used=model_name,
            root_cause_category=root_cause_category,
            causal_chain=causal_chain,
            routing=routing,
            explanation=explanation,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/infer-root-cause", response_model=RootCauseResponse, tags=["diagnostics"])
async def infer_root_cause(request: SymptomRequest):
    """Infer the most likely root cause from a symptom-only log string."""
    try:
        logs = [{"timestamp": "2026-09-28T00:00:00Z", "source": "user", "level": "ERROR", "message": request.log_text}]
        return analyzer.infer_root_cause_from_symptom(logs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/metrics", tags=["diagnostics"])
async def calculate_metrics(predicted: dict, ground_truth: dict):
    """Calculate causal chain metrics."""
    ccr = CausalMetrics.causalchainrecall(predicted, ground_truth)
    occr = CausalMetrics.orderedcausalchainrecall(predicted, ground_truth)

    return {
        "causal_chain_recall": ccr,
        "ordered_causal_chain_recall": occr,
        "improvement": occr / ccr if ccr > 0 else 0.0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
