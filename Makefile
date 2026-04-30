.PHONY: setup data train tune test demo-a demo-b demo-c streamlit eval mlflow-ui clean all

setup:
	pip install -r requirements.txt

data:
	python -m src.generate_data --n 2000 --out data

train: data
	python -m src.train_claims_rf
	python -m src.train_labs_nn

tune:
	python -m src.tune_claims_rf --trials 30 --cv-folds 5

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns

test:
	pytest tests/ -v

demo-a:
	python -m src.main --case A -v

demo-b:
	python -m src.main --case B -v

demo-c:
	python -m src.main --case C -v

streamlit:
	streamlit run streamlit_app.py

eval:
	python -m eval.run_eval --runs 3

clean:
	rm -rf data/ models/ logs/ __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache

all: setup data train test
