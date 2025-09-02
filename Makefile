.PHONY: init 
init:
	uv venv -p python3.12 .venv
	source .venv/bin/activate && uv sync && uv pip freeze > requirements.txt 
