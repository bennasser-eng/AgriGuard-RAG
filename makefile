.PHONY: dataset train test run-all

dataset:
	python -m src.router.generate_dataset

train:
	python -m src.router.train_router

test:
	python -m tests.test_router

train-router-all: dataset train test