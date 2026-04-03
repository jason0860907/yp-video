.PHONY: install dev

install:
	sudo apt-get update && sudo apt-get install -y ffmpeg
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv -p 3.12 --seed
	uv sync
	git clone https://github.com/happyharrycn/actionformer_release.git actionformer_release 2>/dev/null || true
	cd actionformer_release/libs/utils && $(CURDIR)/.venv/bin/python setup.py build_ext --inplace
	cp actionformer_release/libs/utils/nms_1d_cpu*.so $(CURDIR)/.venv/lib/python3.12/site-packages/
	sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8080

dev:
	uv run yp-app
