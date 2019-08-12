run:
	PYTHONPATH=python_modules python cancer_trainer.py

clean:
	rm -r -f python_modules __pycache__

init:
	pip install -r requirements.txt --target=python_modules
