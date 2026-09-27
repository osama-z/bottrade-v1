#!/usr/bin/env bash
# NeuronTrade server bootstrap — Ubuntu 22.04/24.04 VPS.
#
# One command from a fresh server:
#   curl -fsSL https://raw.githubusercontent.com/osama-z/bottrade/version1/deploy/setup_server.sh | bash
# or, after cloning manually:
#   bash deploy/setup_server.sh
#
# Idempotent: safe to re-run for updates (git pull + pip install + unit refresh).

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/osama-z/bottrade.git}"
BRANCH="${BRANCH:-version1}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/neurontrade}"

echo "── NeuronTrade bootstrap ──────────────────────────────"
echo "   repo:    $REPO_URL ($BRANCH)"
echo "   install: $INSTALL_DIR"

# ── 1. System packages ────────────────────────────────────────────────────────
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip curl

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    echo "ERROR: Python >= 3.11 required (found $PYVER)."
    echo "On Ubuntu 22.04: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12 python3.12-venv"
    exit 1
fi

# ── 2. Clone / update ─────────────────────────────────────────────────────────
if [ ! -d "$INSTALL_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
else
    git -C "$INSTALL_DIR" pull origin "$BRANCH"
fi
cd "$INSTALL_DIR"

# ── 3. Virtualenv + dependencies ──────────────────────────────────────────────
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip -q
# Prefer the LOCKED set (exact versions the test suite ran against) for
# reproducible deploys; requirements.txt ranges are the fallback only.
if [ -f requirements.lock ]; then
    .venv/bin/pip install -r requirements.lock -q
else
    .venv/bin/pip install -r requirements.txt -q
fi

# ── 4. Config ─────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    # -m 600: the file will hold API keys — owner-only, never group/world
    install -m 600 .env.example .env
    echo ""
    echo ">>> ACTION REQUIRED: edit $INSTALL_DIR/.env"
    echo ">>>   BINANCE_API_KEY / BINANCE_API_SECRET  (Reading-only key,"
    echo ">>>     IP-whitelisted to THIS server: $(curl -s --max-time 5 ifconfig.me || echo '<server IP>'))"
    echo ">>>   GROQ_API_KEY, NEWS_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID"
fi

# Tighten perms on an .env that already existed (created before this
# script enforced 600, or hand-copied over)
[ -f .env ] && chmod 600 .env

# ── 5. Verify the install ─────────────────────────────────────────────────────
.venv/bin/python -m pytest tests/ -q --no-header 2>&1 | tail -1

# ── 6. systemd units (paths + user templated for this machine) ────────────────
# neurontrade         = single-process paper run (run_live.py) — the default.
# neurontrade-{intelligence,execution} = advanced two-process ZMQ alternative.
# All are installed; you ENABLE one path or the other, never both.
for unit in neurontrade neurontrade-intelligence neurontrade-execution; do
    sed -e "s|/home/osama/ai-workspace/projects/neurontrade|$INSTALL_DIR|g" \
        -e "s|User=osama|User=$USER|g" \
        "deploy/${unit}.service" | sudo tee "/etc/systemd/system/${unit}.service" > /dev/null
done
sudo systemctl daemon-reload

echo ""
echo "── Bootstrap complete ─────────────────────────────────"
echo "Paper run (default — run_live.py: honors STRATEGY, decides on CLOSED candles):"
echo "  1. Edit $INSTALL_DIR/.env  (keys above; STRATEGY=trend_following, DEFAULT_TIMEFRAME=4h)"
echo "  2. Whitelist this server's IP on the Binance key"
echo "  3. .venv/bin/python scripts/health_check.py        # all green?"
echo "  4. sudo systemctl enable --now neurontrade         # start the paper run"
echo "  5. journalctl -fu neurontrade                      # watch it live"
echo "  6. .venv/bin/python scripts/paper_status.py        # one-page status, anytime"
echo "  7. Run the kill-switch drill (deploy/README.md) before leaving it unattended"
echo ""
echo "Only if STRATEGY=ai_combined (needs trained models — trend_following does NOT):"
echo "  .venv/bin/python scripts/train_model.py && .venv/bin/python scripts/train_regime.py"
echo "Advanced two-process ZMQ path — use INSTEAD of neurontrade, never alongside:"
echo "  sudo systemctl enable --now neurontrade-intelligence neurontrade-execution"
