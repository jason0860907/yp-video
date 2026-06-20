.PHONY: install build-ext dev tunnel contract contract-check

VENV ?= $(CURDIR)/.venv
PYTHON ?= $(VENV)/bin/python
PYTHON_VERSION ?= python3.12
SITE_PACKAGES = $(VENV)/lib/$(PYTHON_VERSION)/site-packages

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

dev:
	uv run yp-app

tunnel:
	cloudflared tunnel --url http://localhost:8080

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
