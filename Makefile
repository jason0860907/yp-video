.PHONY: install build-ext dev tunnel

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
