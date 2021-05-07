step0:
	mkdir -p data/junipersun/raw
	mkdir -p data/junipersun/split
	python3 0_junipersun_dataset_acquire_and_split.py

step1:
	python3 1_topological_analysis.py

step2:
	python3 2_feature_analysis.py