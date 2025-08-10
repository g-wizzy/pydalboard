################
####   UV   ####
################

.PHONY: uv-venv
uv-venv:
	uv venv .venv

.PHONY: uv-add
uv-add:
	@bash -c 'if [ -z "$(package)" ]; then echo "Usage: make uv-add package=<name>"; exit 1; fi'
	uv add $(package)

.PHONY: uv-add-dev
uv-add-dev:
	@bash -c 'if [ -z "$(package)" ]; then echo "Usage: make uv-add-dev package=<name>"; exit 1; fi'
	uv add --group dev $(package)

.PHONY: uv-sync
uv-sync:
	uv sync

################
#### PYTHON ####
################

.PHONY: check
check:
	.venv/bin/ruff format -q --check main.py pydalboard
	.venv/bin/ruff check -q main.py pydalboard

.PHONY: fix
fix:
	.venv/bin/ruff format -q main.py pydalboard
	.venv/bin/ruff check -q --fix main.py pydalboard
