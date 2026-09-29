# Developer shortcuts. Run `make help` for the list.
TEST_DB_CONTAINER := docuchat-test-db
TEST_DB_PORT := 55433
DEV_DB_CONTAINER := docuchat-dev-db
DEV_DB_PORT ?= 5432

.DEFAULT_GOAL := help
.PHONY: help up down logs dev-db backend frontend install test-db test lint format eval eval-answers

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

up: ## Build and start the full stack (db + backend + frontend) with Docker
	docker compose up --build -d

down: ## Stop the stack (keeps data volumes)
	docker compose down

logs: ## Follow backend logs
	docker compose logs -f backend

install: ## Install backend (uv) and frontend (npm) dependencies
	cd backend && uv sync
	cd frontend && npm ci

dev-db: ## Start PostgreSQL + pgvector for local development on localhost:$(DEV_DB_PORT)
	docker start $(DEV_DB_CONTAINER) 2>/dev/null || docker run -d --name $(DEV_DB_CONTAINER) \
		-e POSTGRES_USER=docuchat -e POSTGRES_PASSWORD=docuchat -e POSTGRES_DB=docuchat \
		-p 127.0.0.1:$(DEV_DB_PORT):5432 pgvector/pgvector:pg16

backend: ## Run the API with auto-reload (needs a database, see README)
	cd backend && uv run alembic upgrade head && uv run uvicorn app.main:create_app --factory --reload --port 8000

frontend: ## Run the Next.js dev server on :3000
	cd frontend && npm run dev

test-db: ## Start a throwaway pgvector database for the integration tests
	docker start $(TEST_DB_CONTAINER) 2>/dev/null || docker run -d --name $(TEST_DB_CONTAINER) \
		-e POSTGRES_USER=docuchat -e POSTGRES_PASSWORD=docuchat -e POSTGRES_DB=docuchat_test \
		-p 127.0.0.1:$(TEST_DB_PORT):5432 pgvector/pgvector:pg16

test: ## Run backend tests (integration tests are skipped if the test DB is down)
	cd backend && uv run pytest

lint: ## Lint and type-check backend and frontend
	cd backend && uv run ruff check . && uv run ruff format --check . && uv run --extra rerank mypy app scripts
	cd frontend && npm run lint && npx tsc --noEmit

format: ## Auto-format backend code
	cd backend && uv run ruff format . && uv run ruff check --fix .

eval: ## Run the retrieval evaluation inside the running backend container
	docker compose exec backend python scripts/eval.py

eval-answers: ## Run the answer-quality evaluation (JUDGE_PROVIDER) inside the backend container
	docker compose exec backend python scripts/eval_answers.py
