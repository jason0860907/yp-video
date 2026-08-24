.PHONY: install build-web dev db-up db-down serve attach stop contract contract-check

SESSION ?= yp

FRONTEND_DIR ?= src/yp_video/web/frontend
WEB_DIST = $(FRONTEND_DIR)/dist/index.html

install:
	sudo apt-get update && sudo apt-get install -y ffmpeg
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv -p 3.12 --seed
	uv sync
	@echo "Next: fill in ../.env (see ../.env.example), then make db-up."

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

# The audit database. Loopback-only; the app refuses to start without it.
# --env-file on EVERY compose call: it defaults to .env beside the compose
# file, and ours lives one level up with the rest of the config. A bare
# `docker compose ps` fails to interpolate POSTGRES_PASSWORD.
COMPOSE = docker compose --env-file ../.env

db-up:
	$(COMPOSE) up -d
	@$(COMPOSE) ps

db-down:
	$(COMPOSE) down

# 沒有 tunnel target:cloudflared 是常駐的 systemd service(remotely-managed,
# token 在 unit 檔裡,設定全在 Cloudflare dashboard),開機自啟、不隨這個
# tmux session 起落。狀態看 `systemctl status cloudflared`。

# 一鍵把 yp-app 丟進 tmux 背景跑，關掉 SSH 也不會斷。
# 進程若崩潰會留在原視窗(掉回 shell)方便看錯誤、重跑。
# 可重複執行 = 重啟:若已在跑,會先砍掉舊 session 再全新啟動。
serve:
	@# 已在跑就整個砍掉重開
	@tmux kill-session -t $(SESSION) 2>/dev/null && echo "✓ 已關閉舊 tmux session '$(SESSION)'" || true
	@# 再清掉殘留、仍佔著 :8080 的 yp-app(session 被砍但進程沒死),避免新進程綁不到 port
	@if pkill -x yp-app 2>/dev/null; then \
		echo "✓ 已關閉殘留的 yp-app"; \
		for i in 1 2 3 4 5; do pgrep -x yp-app >/dev/null 2>&1 || break; sleep 1; done; \
	fi
	@tmux new-session -d -s $(SESSION) -n dev '$(MAKE) dev; exec $$SHELL'
	@echo "✓ 已在 tmux session '$(SESSION)' 啟動 yp-app"
	@echo "   https://label.volley-iq.com  公開網址(Cloudflare Access 登入)"
	@echo "   make attach  進去看畫面(離開按 Ctrl-b 再按 d)"
	@echo "   make stop    關閉整個服務"

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
	uv run python -m yp_video.web.make_train_schemas

# CI / pre-commit guard against drift: regenerate and fail if the committed
# schemas are stale (someone edited a Pydantic model but forgot `make contract`).
contract-check:
	uv run python -m yp_video.contracts.make_schema
	uv run python -m yp_video.web.make_train_schemas
	@git diff --exit-code -- contracts/*.schema.json \
		|| (echo "❌ contracts/*.schema.json out of date — run 'make contract' and commit." && exit 1)
	@echo "✓ contracts up to date"
