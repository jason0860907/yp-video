.PHONY: install build-ext build-web dev tunnel serve attach url stop contract contract-check

SESSION ?= yp

VENV ?= $(CURDIR)/.venv
PYTHON ?= $(VENV)/bin/python
PYTHON_VERSION ?= python3.12
SITE_PACKAGES = $(VENV)/lib/$(PYTHON_VERSION)/site-packages

FRONTEND_DIR ?= src/yp_video/web/frontend
WEB_DIST = $(FRONTEND_DIR)/dist/index.html

install:
	sudo apt-get update && sudo apt-get install -y ffmpeg
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv -p 3.12 --seed
	uv sync
	$(MAKE) build-ext
	sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8080

# Build the ActionFormer NMS C extension into the target venv.
# External venvs can call this via: make build-ext VENV=/path/to/other/.venv
build-ext:
	cd actionformer/libs/utils && $(PYTHON) setup.py build_ext --inplace
	cp actionformer/libs/utils/nms_1d_cpu*.so $(SITE_PACKAGES)/

# Build the React SPA that FastAPI serves at :8080. Rebuilds only when the
# frontend sources change (make compares against dist/index.html).
$(FRONTEND_DIR)/node_modules: $(FRONTEND_DIR)/package.json
	cd $(FRONTEND_DIR) && npm install
	@touch $@

$(WEB_DIST): $(FRONTEND_DIR)/node_modules $(shell find $(FRONTEND_DIR)/src $(FRONTEND_DIR)/index.html $(FRONTEND_DIR)/tailwind.config.js $(FRONTEND_DIR)/vite.config.ts -type f 2>/dev/null)
	cd $(FRONTEND_DIR) && npm run build

build-web: $(WEB_DIST)

dev: build-web
	uv run yp-app

tunnel:
	cloudflared tunnel --url http://localhost:8080

# 一鍵把 dev + tunnel 丟進 tmux 背景跑，關掉 SSH 也不會斷。
# 進程若崩潰會留在原視窗(掉回 shell)方便看錯誤、重跑。
serve:
	@if tmux has-session -t $(SESSION) 2>/dev/null; then \
		echo "⚠️  tmux session '$(SESSION)' 已在跑 — 用 make url / make attach / make stop"; \
		exit 1; \
	fi
	@tmux new-session -d -s $(SESSION) -n dev    '$(MAKE) dev;    exec $$SHELL'
	@tmux new-window  -t $(SESSION)   -n tunnel  '$(MAKE) tunnel; exec $$SHELL'
	@echo "✓ 已在 tmux session '$(SESSION)' 啟動 dev + tunnel"
	@echo "   make url     取得公開網址(等幾秒讓 tunnel 起來)"
	@echo "   make attach  進去看畫面(離開按 Ctrl-b 再按 d)"
	@echo "   make stop    關閉整個服務"

# 從 tunnel 視窗的輸出撈出 trycloudflare 公開網址。
url:
	@u=$$(tmux capture-pane -t $(SESSION):tunnel -p -S - 2>/dev/null \
		| grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1); \
	if [ -n "$$u" ]; then echo "$$u"; \
	else echo "⏳ 還抓不到網址 — tunnel 可能還在啟動,等幾秒再 make url(或 make attach 看 tunnel 視窗)"; fi

attach:
	@tmux attach -t $(SESSION)

stop:
	@tmux kill-session -t $(SESSION) 2>/dev/null \
		&& echo "✓ 已關閉 tmux session '$(SESSION)'" \
		|| echo "session '$(SESSION)' 沒在跑"

# Regenerate the JSON-schema contracts from the Pydantic models. Run after
# editing contracts/*.py. The emitted contracts/*.schema.json are the source
# of truth consumed by the iOS app + yp-spot.
contract:
	uv run python -m yp_video.contracts.make_schema

# CI / pre-commit guard against drift: regenerate and fail if the committed
# schemas are stale (someone edited a Pydantic model but forgot `make contract`).
contract-check:
	uv run python -m yp_video.contracts.make_schema
	@git diff --exit-code -- contracts/*.schema.json \
		|| (echo "❌ contracts/*.schema.json out of date — run 'make contract' and commit." && exit 1)
	@echo "✓ contracts up to date"
