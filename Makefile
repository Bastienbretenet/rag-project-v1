build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

rebuild:
	docker compose build && docker compose up -d
