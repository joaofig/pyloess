BIN = ./venv/bin/
PYTHON = $(BIN)python
UV = ./venv/bin/uv

install:
	pyenv install --skip-existing
	pyenv exec python -m venv venv
	$(BIN)pip install --upgrade pip
	$(BIN)pip install uv
	$(UV) pip sync requirements/run.txt

uninstall:
	rm -rf venv

compile:
	$(UV) pip compile requirements/run.in -o requirements/run.txt

sync:
	$(UV) pip sync requirements/run.txt
