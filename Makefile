define compile-requirements =
pip-compile $< --generate-hashes -o $@
endef

.PHONY: data requirements

# Download dataset.
data:
	mkdir -p data
	wget -nc https://cvg.ethz.ch/research/symmetries-in-sfm/datasets/barcelona.zip -P data
	unzip -n data/barcelona.zip -d data/barcelona

requirements.txt: requirements.in
	${compile-requirements}

requirements_ci.txt: requirements_ci.in
	${compile-requirements}

requirements: requirements.txt requirements_ci.txt
