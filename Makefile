.PHONY: help install env fmt lint test test-unit test-k run smoke-imports

MODEL ?= models/all_weights/idd_dataset.pt
VIDEO ?= test/test_videos/video1.mp4
INTERVAL ?= 2
# Space-separated list of class ids
CLASS_IDS ?= 0 1 2 3 4

help:
	@echo "Available targets:"
	@echo "  install        - Install dependencies with Poetry"
	@echo "  env            - Show Poetry env info and key package versions"
	@echo "  fmt            - Format code (black + isort)"
	@echo "  lint           - Lint code (flake8)"
	@echo "  test           - Run all tests (pytest)"
	@echo "  test-unit      - Run unit tests (alias of test)"
	@echo "  test-k K=expr  - Run tests filtered by -k expression"
	@echo "  run            - Run main pipeline (customizable via MODEL/VIDEO/INTERVAL/CLASS_IDS)"
	@echo "  smoke-imports  - Quick import check for core modules"

install:
	poetry install

env:
	poetry env info
	poetry run python -c "import sys; print('Python:', sys.version)"
	poetry run python -c "import ultralytics, supervision, yolox; print('ultralytics', ultralytics.__version__, '| supervision', supervision.__version__, '| yolox', getattr(yolox,'__version__','n/a'))" || true

fmt:
	poetry run isort .
	poetry run black .

lint:
	poetry run flake8 .

test:
	poetry run pytest -q

test-unit: test

test-k:
	@# Usage: make test-k K="config and results"
	poetry run pytest -q -k "$(K)"

run:
	poetry run python main.py \
		--model_yolo $(MODEL) \
		--source_video $(VIDEO) \
		--interval_time $(INTERVAL) \
		--class_ids $(CLASS_IDS)

smoke-imports:
	poetry run python - <<'PY'
from traffic_od.config import AppConfig
from traffic_od.detector import YoloV8Detector
from traffic_od.tracker import ByteTrackAdapter, TrackerConfig
from traffic_od.pipeline import VideoPipeline
print('Core module imports: OK')
PY
