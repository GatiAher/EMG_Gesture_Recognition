step0:
	mkdir -p data/junipersun/raw
	mkdir -p data/junipersun/split
	python3 0_junipersun_dataset_acquire_and_split.py

step1:
	mkdir -p data/junipersun/TDA_data
	python3 1_feature_extraction_TDA.py
	python3 1_TDA.py

step2:
	python3 2_feature_analysis.py