#!/bin/bash
# LazyMoE Launcher — macOS / Linux

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}"
echo "  ██╗      █████╗ ███████╗██╗   ██╗      ███╗   ███╗ ██████╗ ███████╗"
echo "  ██║     ██╔══██╗╚════██║╚██╗ ██╔╝      ████╗ ████║██╔═══██╗██╔════╝"
echo "  ██║     ███████║    ██╔╝  ╚████╔╝       ██╔████╔██║██║   ██║█████╗  "
echo "  ██║     ██╔══██║   ██╔╝    ╚██╔╝        ██║╚██╔╝██║██║   ██║██╔══╝  "
echo "  ███████╗██║  ██║   ██║      ██║         ██║ ╚═╝ ██║╚██████╔╝███████╗"
echo "  ╚══════╝╚═╝  ╚═╝   ╚═╝      ╚═╝         ╚═╝     ╚═╝ ╚═════╝ ╚══════╝"
echo -e "${NC}"
echo "  Local LLM Inference  |  LazyMoE v0.3"
echo "  -----------------------------------------------"

# ── Config — edit these ──────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${LAZY_MOE_MODEL:-$ROOT/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf}"
THREADS="${LAZY_MOE_THREADS:-4}"
RAM="${LAZY_MOE_RAM_GB:-8}"

export LAZY_MOE_MODEL="$MODEL"
export LAZY_MOE_THREADS="$THREADS"
export LAZY_MOE_RAM_GB="$RAM"
export PATH="$ROOT/llama.cpp:$PATH"

# ── Check deps ───────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then echo -e "${RED}Python3 not found${NC}"; exit 1; fi
if ! command -v npm &>/dev/null; then echo -e "${RED}npm not found${NC}"; exit 1; fi

# ── Kill any existing servers ─────────────────────────────────────────────────
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

# ── Start backend ─────────────────────────────────────────────────────────────
echo -e "${GREEN}[1/2] Starting backend...${NC}"
cd "$ROOT/backend" && python3 server.py &
BACKEND_PID=$!

sleep 2

# ── Start frontend ────────────────────────────────────────────────────────────
echo -e "${GREEN}[2/2] Starting frontend...${NC}"
cd "$ROOT/frontend" && npm run dev &
FRONTEND_PID=$!

sleep 3

# ── Open browser ──────────────────────────────────────────────────────────────
if command -v open &>/dev/null; then
    open http://localhost:5173
elif command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:5173
fi

echo ""
echo -e "${CYAN}LazyMoE is running!${NC}"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "  Press Ctrl+C to stop all servers"
echo ""

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
