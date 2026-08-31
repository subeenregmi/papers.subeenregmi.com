VAULT ?= $(HOME)/subeenfiles/research/research

# Copy the vault into content: markdown and images only. The PDFs, corpora and
# notebooks stay out because this repo is public and they are large. The
# leading exclude also protects our own index.md from --delete.
sync:
	@rsync -am --delete \
		--exclude="/index.md" \
		--exclude=".obsidian/" --exclude=".ipynb_checkpoints/" --exclude=".DS_Store" \
		--include="*/" --include="*.md" --include="*.png" --include="*.jpg" \
		--include="*.jpeg" --include="*.gif" --include="*.svg" --exclude="*" \
		"$(VAULT)/" content/
	@git status --short content

build:
	@docker compose build

up:
	@docker compose up -d

down:
	@docker compose down

# Ctrl-C leaves 130. Anything else is a real failure and must still stop make.
logs:
	@docker compose logs -f || [ $$? -eq 130 ]

# Forced on the deploy key in the servers authorized_keys.
deploy:
	git fetch origin v5
	git reset --hard origin/v5
	@docker compose up -d --build
