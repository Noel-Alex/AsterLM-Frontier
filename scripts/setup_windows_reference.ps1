$ErrorActionPreference = "Stop"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch
python -m pip install -e ".[dev]"
python scripts\system_check.py --model configs\model\aster_110m.yaml
python -m pytest
Write-Host "Reference environment ready. Use WSL2/Linux plus fla-core for serious CUDA training."
