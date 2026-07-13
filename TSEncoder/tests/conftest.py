"""Put the TSEncoder package root (this dir's parent) on sys.path so tests can
`import ditar_encoder` without an install or PYTHONPATH."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
