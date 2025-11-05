.PHONY: help preprocess index qa eval clean install test test-all test-integration test-coverage test-quick setup-uv

# デフォルト設定
PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin
DATA_DIR := data
DATASETS_DIR := datasets
INDEX_DIR := $(DATA_DIR)/faiss_index
JSONL_FILE := $(DATA_DIR)/egov_laws.jsonl
LAWQA_DATASET := $(DATASETS_DIR)/lawqa_jp/data/selection.json
EVAL_REPORT := $(DATA_DIR)/evaluation_report.json

# 処理制限（テスト用）
PREPROCESS_LIMIT ?= 
INDEX_LIMIT ?=
EVAL_LIMIT ?= 10

help:
	@echo "Legal RAG System - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make setup-uv              - uvで仮想環境をセットアップ"
	@echo "  make install               - 依存パッケージをインストール"
	@echo "  make install-examples      - Examples用依存関係をインストール"
	@echo "  make preprocess            - XML→JSONL前処理"
	@echo "  make index                 - ベクトルインデックス構築"
	@echo "  make qa                    - 対話型CLI起動"
	@echo "  make eval                  - RAGAS評価実行"
	@echo "  make eval-multiple-choice  - 4択法令データ評価実行"
	@echo "  make eval-agentic-rag      - Agentic RAG評価実行"
	@echo "  make test                  - ユニットテスト実行"
	@echo "  make test-all              - 全テスト実行"
	@echo "  make test-examples         - Examplesテスト実行"
	@echo "  make test-coverage         - カバレッジ付きテスト実行"
	@echo "  make clean                 - 生成ファイル削除"
	@echo "  make all                   - 全ステップ実行 (preprocess → index)"
	@echo ""
	@echo "Environment variables:"
	@echo "  PREPROCESS_LIMIT   - 前処理ファイル数制限 (デフォルト: 無制限)"
	@echo "  INDEX_LIMIT        - インデックス化ドキュメント数制限 (デフォルト: 無制限)"
	@echo "  EVAL_LIMIT         - 評価サンプル数制限 (デフォルト: 10)"
	@echo ""
	@echo "Examples:"
	@echo "  make preprocess PREPROCESS_LIMIT=100"
	@echo "  make index INDEX_LIMIT=1000"
	@echo "  make eval EVAL_LIMIT=50"
	@echo "  make eval-multiple-choice EVAL_LIMIT=20"
	@echo "  make eval-agentic-rag"

setup-uv:
	@echo "Setting up environment with uv..."
	./setup/setup_uv_env.sh
	@echo "Setup complete!"

install:
	@echo "Installing dependencies..."
	@echo "Note: For initial setup, use './setup/setup_uv_env.sh' instead."
	@echo "This target is for reinstalling dependencies in an existing environment."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	@echo "SudachiPy and Janome are now included in dependencies (no admin rights required)"
	@echo "Installation complete!"

preprocess:
	@echo "Preprocessing e-Gov XML files to JSONL..."
	@mkdir -p $(DATA_DIR)
	$(PYTHON) scripts/preprocess_egov_xml.py \
		--input-dir $(DATASETS_DIR)/egov_laws \
		--output-file $(JSONL_FILE) \
		$(if $(PREPROCESS_LIMIT),--limit $(PREPROCESS_LIMIT))
	@echo "Preprocessing complete! Output: $(JSONL_FILE)"

index:
	@echo "Building vector index..."
	@mkdir -p $(INDEX_DIR)
	$(PYTHON) scripts/build_index.py \
		--data-path $(JSONL_FILE) \
		--index-path $(INDEX_DIR) \
		$(if $(INDEX_LIMIT),--limit $(INDEX_LIMIT))
	@echo "Index built successfully! Location: $(INDEX_DIR)"

qa:
	@echo "Starting interactive QA CLI..."
	$(PYTHON) scripts/query_cli.py --interactive

query:
	@if [ -z "$(Q)" ]; then \
		echo "Error: Please provide a question with Q='your question'"; \
		exit 1; \
	fi
	@echo "Question: $(Q)"
	@$(PYTHON) scripts/query_cli.py "$(Q)"

eval:
	@echo "Running RAGAS evaluation..."
	@mkdir -p $(DATA_DIR)
	$(PYTHON) scripts/evaluate_ragas.py \
		--dataset $(LAWQA_DATASET) \
		--output $(EVAL_REPORT) \
		--limit $(EVAL_LIMIT)
	@echo "Evaluation complete! Report: $(EVAL_REPORT)"

eval-multiple-choice:
	@echo "Running 4-choice evaluation..."
	@mkdir -p $(DATA_DIR)
	$(PYTHON) scripts/evaluate_multiple_choice.py \
		--data $(LAWQA_DATASET) \
		--output evaluation_results.json \
		--samples $(EVAL_LIMIT)
	@echo "4-choice evaluation complete! Report: evaluation_results.json"

test:
	@echo "Running unit tests..."
	./scripts/run_tests.sh unit

test-all:
	@echo "Running all tests..."
	./scripts/run_tests.sh all

test-integration:
	@echo "Running integration tests..."
	./scripts/run_tests.sh integration

test-coverage:
	@echo "Running tests with coverage..."
	./scripts/run_tests.sh coverage

test-quick:
	@echo "Running quick tests..."
	./scripts/run_tests.sh quick

clean:
	@echo "Cleaning generated files..."
	rm -rf $(DATA_DIR)/*.jsonl
	rm -rf $(INDEX_DIR)
	rm -rf $(DATA_DIR)/evaluation_report.json
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"

all: preprocess index
	@echo "All steps completed successfully!"

# 実験用ターゲット
experiment-small: clean
	@echo "Running small-scale experiment..."
	$(MAKE) preprocess PREPROCESS_LIMIT=10
	$(MAKE) index INDEX_LIMIT=100
	$(MAKE) eval EVAL_LIMIT=5
	@echo "Small experiment complete!"

experiment-medium: clean
	@echo "Running medium-scale experiment..."
	$(MAKE) preprocess PREPROCESS_LIMIT=100
	$(MAKE) index INDEX_LIMIT=1000
	$(MAKE) eval EVAL_LIMIT=20
	@echo "Medium experiment complete!"

# 開発用ターゲット
dev-setup: install
	@echo "Setting up development environment..."
	$(PYTHON) -m pip install black ruff pytest pytest-cov
	@echo "Dev setup complete!"

lint:
	@echo "Running linters..."
	black --check app/ scripts/
	ruff check app/ scripts/

format:
	@echo "Formatting code..."
	black app/ scripts/
	@echo "Format complete!"

# Examples用ターゲット
install-examples:
	@echo "Installing examples dependencies..."
	./setup/setup_examples.sh
	@echo "Examples dependencies installed!"

test-examples:
	@echo "Running examples tests..."
	cd examples && pytest tests/test_01_agentic_rag.py -v -m "not integration"
	@echo "Examples tests complete!"

eval-agentic-rag:
	@echo "Running Agentic RAG evaluation (limited)..."
	cd examples/01_agentic_rag && python evaluate.py \
		--dataset ../../datasets/lawqa_jp/data/selection.json \
		--output results/eval_$(shell date +%Y%m%d_%H%M%S).json \
		--max-questions 5
	@echo "Agentic RAG evaluation complete!"
