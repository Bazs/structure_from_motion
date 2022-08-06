define compile-requirements =
pip-compile $< --generate-hashes -o $@
endef

# Download dataset.
data:
	mkdir -p data
	wget -nc https://cvg.ethz.ch/research/symmetries-in-sfm/datasets/barcelona.zip -P data
	unzip -n data/barcelona.zip -d data/barcelona

.PHONY: requirements.txt
requirements.txt: requirements.in
	${compile-requirements}

.PHONY: requirements_ci.txt
requirements_ci.txt: requirements_ci.in
	${compile-requirements}

.PHONY: requirements
requirements: requirements.txt requirements_ci.txt
