.PHONY: test demo

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 -m relational_lab.analysis examples/toy_counterfactual.json
