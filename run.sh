#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  run.sh — Cephalometric Landmark Detection launcher
#
#  Usage:
#    ./run.sh setup            — Create venv & install dependencies
#    ./run.sh train [args...]  — Run model training (uses MPS on Mac)
#    ./run.sh docker build     — Build Docker image
#    ./run.sh docker start     — Run app inside Docker container
#    ./run.sh docker stop      — Stop and remove the container
#    ./run.sh local start      — Run app in local .venv
#    ./run.sh local stop       — Kill the local Streamlit process
#    ./run.sh mlflow start     — Launch MLflow UI (experiments viewer)
#    ./run.sh mlflow stop      — Kill the MLflow UI process
#
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ───────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_IMAGE="cephalometric-demo:0.1"
DOCKER_CONTAINER="cephalometric-demo"
PORT=8501
VENV_DIR="${PROJECT_DIR}/.venv"
APP_FILE="${PROJECT_DIR}/app/streamlit_app.py"
PID_FILE="/tmp/cephalometric-local.pid"
MLFLOW_DIR="${PROJECT_DIR}/mlruns"
MLFLOW_PORT=5000
MLFLOW_PID_FILE="/tmp/cephalometric-mlflow.pid"

# ── Helpers ──────────────────────────────────────────────────
info()    { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
success() { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

usage() {
    echo ""
    echo "  Usage: ./run.sh <mode> [action] [args...]"
    echo ""
    echo "  Modes:"
    echo "    setup              — Create .venv & install dependencies"
    echo "    train [args...]    — Run model training (MPS-accelerated on Mac)"
    echo "    docker build|start|stop"
    echo "    local  start|stop  — Run directly with local .venv"
    echo "    mlflow start|stop  — Launch MLflow experiment tracking UI"
    echo ""
    echo "  Examples:"
    echo "    ./run.sh setup"
    echo "    ./run.sh train --epochs 50 --lr 1e-3"
    echo "    ./run.sh docker build"
    echo "    ./run.sh docker start"
    echo "    ./run.sh docker stop"
    echo "    ./run.sh local start"
    echo "    ./run.sh local stop"
    echo "    ./run.sh mlflow start"
    echo "    ./run.sh mlflow stop"
    echo ""
    exit 1
}

# ── Setup ────────────────────────────────────────────────────
do_setup() {
    info "Setting up project environment..."

    # Create project directories
    for dir in data checkpoints mlruns sample_images; do
        mkdir -p "${PROJECT_DIR}/${dir}"
        success "Directory: ${dir}/"
    done

    # Create virtual environment
    if [ ! -f "${VENV_DIR}/bin/python" ]; then
        info "Creating virtual environment with $(python3.12 --version 2>&1)..."
        python3.12 -m venv "${VENV_DIR}"
        success "Virtual environment created at .venv/"
    else
        warn "Virtual environment already exists at .venv/"
    fi

    # Install dependencies
    info "Installing dependencies from requirements.txt..."
    "${VENV_DIR}/bin/pip" install --upgrade pip > /dev/null
    "${VENV_DIR}/bin/pip" install -r "${PROJECT_DIR}/requirements.txt"
    success "All dependencies installed."

    echo ""
    success "Setup complete! Next steps:"
    info "  1. Place your data in data/"
    info "  2. Run training:  ./run.sh train"
    info "  3. Launch app:    ./run.sh local start"
}

# ── Train ────────────────────────────────────────────────────
do_train() {
    shift  # remove 'train' from args, pass the rest through

    if [ ! -f "${VENV_DIR}/bin/python" ]; then
        error "Virtual environment not found at '.venv'.\nRun setup first:\n  ./run.sh setup"
    fi

    # Detect MPS (Apple Silicon GPU)
    info "Checking hardware acceleration..."
    MPS_AVAILABLE=$("${VENV_DIR}/bin/python" -c "
import torch
print('yes' if torch.backends.mps.is_available() else 'no')
" 2>/dev/null || echo "no")

    if [ "$MPS_AVAILABLE" = "yes" ]; then
        success "Apple MPS (Metal) GPU backend detected — training will be GPU-accelerated."
    else
        warn "MPS not available — training will use CPU."
    fi

    info "Starting training..."
    "${VENV_DIR}/bin/python" "${PROJECT_DIR}/train.py" "$@"
}

# ── Docker mode ──────────────────────────────────────────────
docker_build() {
    info "Checking Docker..."
    command -v docker >/dev/null 2>&1 || error "Docker is not installed or not in PATH."

    info "Building Docker image '$DOCKER_IMAGE'..."
    docker build -t "$DOCKER_IMAGE" "$PROJECT_DIR"
    success "Docker image '$DOCKER_IMAGE' built successfully."
}

docker_start() {
    info "Checking Docker..."
    command -v docker >/dev/null 2>&1 || error "Docker is not installed or not in PATH."

    # Check image exists
    if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        error "Docker image '$DOCKER_IMAGE' not found. Build it first:\n  ./run.sh docker build"
    fi

    # Remove stale container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER}$"; then
        warn "Container '$DOCKER_CONTAINER' already exists — removing it first."
        docker rm -f "$DOCKER_CONTAINER" >/dev/null
    fi

    info "Starting Docker container '$DOCKER_CONTAINER' on port $PORT..."
    docker run -d \
        --name "$DOCKER_CONTAINER" \
        -p "${PORT}:8501" \
        "$DOCKER_IMAGE" >/dev/null

    # Wait for health check
    info "Waiting for Streamlit to be ready..."
    for i in {1..20}; do
        if curl -sf "http://localhost:${PORT}/_stcore/health" >/dev/null 2>&1; then
            success "App is running → http://localhost:${PORT}"
            return 0
        fi
        sleep 1
    done
    warn "Health check timed out — container may still be starting."
    info "Check logs with: docker logs $DOCKER_CONTAINER"
}

docker_stop() {
    if docker ps --format '{{.Names}}' | grep -q "^${DOCKER_CONTAINER}$"; then
        info "Stopping container '$DOCKER_CONTAINER'..."
        docker stop "$DOCKER_CONTAINER" >/dev/null
        docker rm "$DOCKER_CONTAINER" >/dev/null
        success "Container stopped and removed."
    else
        warn "No running container named '$DOCKER_CONTAINER' found."
    fi
}

# ── Local mode ───────────────────────────────────────────────
local_start() {
    # Validate venv
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        error "Virtual environment not found at '.venv'.\nCreate it first:\n  ./run.sh setup"
    fi

    # Validate app file
    if [ ! -f "$APP_FILE" ]; then
        error "streamlit_app.py not found at '$APP_FILE'."
    fi

    # Check if already running
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        warn "Streamlit already running (PID $(cat "$PID_FILE"))."
        info "  Stop it first with: ./run.sh local stop"
        exit 1
    fi

    info "Starting Streamlit with local .venv on port $PORT..."
    nohup "$VENV_DIR/bin/streamlit" run "$APP_FILE" \
        --server.port="$PORT" \
        --server.headless=true \
        --browser.gatherUsageStats=false \
        > /tmp/cephalometric-local.log 2>&1 &

    echo $! > "$PID_FILE"
    info "Waiting for Streamlit to be ready..."

    for i in {1..20}; do
        if curl -sf "http://localhost:${PORT}/_stcore/health" >/dev/null 2>&1; then
            success "App is running → http://localhost:${PORT}  (PID $(cat "$PID_FILE"))"
            info "  Logs: tail -f /tmp/cephalometric-local.log"
            return 0
        fi
        sleep 1
    done
    warn "Health check timed out — check logs: tail -f /tmp/cephalometric-local.log"
}

local_stop() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        PID=$(cat "$PID_FILE")
        info "Stopping Streamlit (PID $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        success "Streamlit stopped."
    else
        # Try finding it by port as fallback
        PID=$(lsof -ti tcp:"$PORT" 2>/dev/null | head -1 || true)
        if [ -n "$PID" ]; then
            info "Stopping process on port $PORT (PID $PID)..."
            kill "$PID"
            rm -f "$PID_FILE"
            success "Stopped."
        else
            warn "No local Streamlit process found on port $PORT."
        fi
    fi
}

# ── MLflow mode ──────────────────────────────────────────────
mlflow_start() {
    if [ ! -d "$MLFLOW_DIR" ]; then
        mkdir -p "$MLFLOW_DIR"
        info "Created mlruns/ directory."
    fi

    if [ ! -f "$VENV_DIR/bin/mlflow" ]; then
        error "mlflow not found in venv. Run: ./run.sh setup"
    fi

    if [ -f "$MLFLOW_PID_FILE" ] && kill -0 "$(cat "$MLFLOW_PID_FILE")" 2>/dev/null; then
        warn "MLflow UI already running (PID $(cat "$MLFLOW_PID_FILE"))."
        info "  Stop it first with: ./run.sh mlflow stop"
        exit 1
    fi

    info "Starting MLflow UI on port $MLFLOW_PORT..."
    nohup "$VENV_DIR/bin/mlflow" ui \
        --backend-store-uri "sqlite:///${PROJECT_DIR}/mlflow.db" \
        --port "$MLFLOW_PORT" \
        > /tmp/cephalometric-mlflow.log 2>&1 &

    echo $! > "$MLFLOW_PID_FILE"
    info "Waiting for MLflow UI to be ready..."

    # Any HTTP response (even 4xx) means the server is up and accepting connections
    for i in {1..30}; do
        if curl -s -o /dev/null "http://localhost:${MLFLOW_PORT}" >/dev/null 2>&1; then
            success "MLflow UI running → http://localhost:${MLFLOW_PORT}  (PID $(cat "$MLFLOW_PID_FILE"))"
            info "  Logs: tail -f /tmp/cephalometric-mlflow.log"
            return 0
        fi
        sleep 1
    done
    warn "Health check timed out — check logs: tail -f /tmp/cephalometric-mlflow.log"
}

mlflow_stop() {
    if [ -f "$MLFLOW_PID_FILE" ] && kill -0 "$(cat "$MLFLOW_PID_FILE")" 2>/dev/null; then
        PID=$(cat "$MLFLOW_PID_FILE")
        info "Stopping MLflow UI (PID $PID)..."
        kill "$PID"
        rm -f "$MLFLOW_PID_FILE"
        success "MLflow UI stopped."
    else
        PID=$(lsof -ti tcp:"$MLFLOW_PORT" 2>/dev/null | head -1 || true)
        if [ -n "$PID" ]; then
            info "Stopping process on port $MLFLOW_PORT (PID $PID)..."
            kill "$PID"
            rm -f "$MLFLOW_PID_FILE"
            success "Stopped."
        else
            warn "No MLflow UI process found on port $MLFLOW_PORT."
        fi
    fi
}

# ── Entrypoint ───────────────────────────────────────────────
MODE="${1:-}"

[ -z "$MODE" ] && usage

case "$MODE" in
    setup)
        do_setup
        ;;
    train)
        do_train "$@"
        ;;
    docker)
        ACTION="${2:-}"
        [ -z "$ACTION" ] && usage
        case "$ACTION" in
            build) docker_build ;;
            start) docker_start ;;
            stop)  docker_stop  ;;
            *)     usage ;;
        esac
        ;;
    local)
        ACTION="${2:-}"
        [ -z "$ACTION" ] && usage
        case "$ACTION" in
            start) local_start ;;
            stop)  local_stop  ;;
            *)     usage ;;
        esac
        ;;
    mlflow)
        ACTION="${2:-}"
        [ -z "$ACTION" ] && usage
        case "$ACTION" in
            start) mlflow_start ;;
            stop)  mlflow_stop  ;;
            *)     usage ;;
        esac
        ;;
    *)
        usage
        ;;
esac
