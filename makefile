# Makefile

.PHONY: news all help

help:
	@echo "Comandos disponibles:"
	@echo "  make news    — Corre solo el scraper de noticias"
	@echo "  make all     — Corre todo el pipeline (news + demás módulos)"

news:
	python pipeline.py news

summaries:
	python pipeline.py summaries

summary_vectorization:
	python pipeline.py summary_vectorization

clusterization:
	python pipeline.py clusterization

all:
	python pipeline.py all