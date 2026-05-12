#!/bin/bash
set -e
cd "$(dirname "$0")"

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║   Loan Clarity — Student Loan Sentiment Engine       ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "  ⚠️  ANTHROPIC_API_KEY is not set."
  echo "  ➜  Export it first:  export ANTHROPIC_API_KEY=sk-ant-..."
  echo ""
  read -r -p "  Or paste your key now: " key
  export ANTHROPIC_API_KEY="$key"
fi

echo "  📦 Installing Python dependencies..."
pip install -q -r requirements.txt

echo "  🚀 Starting sentiment engine on http://localhost:8000"
echo "  📊 Open your browser to: http://localhost:8000"
echo "  ⟳  Data refreshes every 5 minutes automatically."
echo ""

# Open browser after 2s
(sleep 2 && open http://localhost:8000) &

python3 app.py
