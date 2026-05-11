# Local aliases for the CNR trajectory generation project.
# Usage: source aliases.sh

alias cnr-root='cd /home/irisliu/Thesis'
alias cnr-activate='source /home/irisliu/Thesis/.venv/bin/activate'
alias cnr-test='cd /home/irisliu/Thesis && PYTHONPATH=src .venv/bin/python -m pytest tests'
alias cnr-compile='cd /home/irisliu/Thesis && .venv/bin/python -m compileall src scripts tests'
