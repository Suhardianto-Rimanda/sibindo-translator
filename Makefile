.PHONY: help install run test clean collect train eval bench docker docker-up docker-down \
        extract-videos extract-photos train-letter

help:
	@echo "Available targets:"
	@echo "  install           Install Python dependencies"
	@echo "  run               Run Flask dev server"
	@echo "  test              Run pytest suite"
	@echo "  collect LBL=x     Collect landmarks via webcam for label x"
	@echo "  inspect           Inspect dataset class distribution"
	@echo "  extract-videos    Extract landmarks from data/raw/videos/ -> data/processed/words/"
	@echo "  extract-photos    Extract landmarks from data/raw/photos/ -> data/processed/letters/"
	@echo "  train             Train LSTM word model (data/processed/words/ or data/processed/)"
	@echo "  train-letter      Train MLP letter model (data/processed/letters/)"
	@echo "  eval              Evaluate LSTM (confusion matrix etc.)"
	@echo "  bench             Run latency benchmark"
	@echo "  docker            Build docker image"
	@echo "  docker-up         Start docker compose stack"
	@echo "  docker-down       Stop docker compose stack"
	@echo "  clean             Clean caches + audio output"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

run:
	python app.py

test:
	pytest -v tests/

collect:
	python scripts/collect_landmarks.py --label $(LBL) --samples 30 --frames 30

inspect:
	python scripts/inspect_dataset.py

extract-videos:
	python scripts/extract_landmarks_from_video.py

extract-photos:
	python scripts/extract_landmarks_from_photo.py

train:
	python scripts/train_lstm.py --epochs 100 --batch 16 --data data/processed/words

train-letter:
	python scripts/train_letter_classifier.py --epochs 100 --batch 32

eval:
	python scripts/evaluate_lstm.py

bench:
	python scripts/benchmark_latency.py --frames 200

docker:
	docker build -t sibindo-translator .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find app/static/audio -name "*.mp3" -delete 2>/dev/null || true
