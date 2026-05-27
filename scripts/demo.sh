#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
WEB_URL="${WEB_URL:-http://localhost:5173}"

echo "== ContextForge demo helper =="
echo "API: ${API_URL}"
echo "WEB: ${WEB_URL}"
echo

echo "[1/5] Checking API health..."
curl -fsS "${API_URL}/v1/health" >/dev/null
echo "OK: API is reachable"
echo

echo "[2/5] Reminder: seed sample corpus if needed"
echo "Run: just seed"
echo

echo "[3/5] Open chat UI"
echo "Open: ${WEB_URL}"
echo "Keep Debug panel enabled."
echo

echo "[4/5] Suggested prompts"
echo "- direct:     hi"
echo "- single-hop: How many PTO days do full-time employees accrue per year?"
echo "- multi-hop:  Compare PTO policy steps with remote work approval steps"
echo "- abstain:    What is the lunar habitat budget for 2099?"
echo

echo "[5/5] Optional eval runs"
echo "- just eval-dry"
echo "- just eval-heuristic"
echo "- just eval   # requires OPENAI_API_KEY"
echo

echo "Demo checklist complete."
