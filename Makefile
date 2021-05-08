step0:
	mkdir -p data/junipersun/raw
	mkdir -p data/junipersun/split
	python3 0_junipersun_dataset_acquire_and_split.py

step1:
	mkdir -p data/junipersun/TDA_data
	python3 1_feature_extraction_TDA.py
	mkdir -p data/junipersun/TDA_results
	python3 1_TDA.py

step2:
	mkdir -p data/junipersun/FA_data
	python3 2_feature_extraction_FA.py
	mkdir -p data/junipersun/FA_results
	python3 2_FA.py