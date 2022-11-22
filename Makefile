define compile-requirements =
pip-compile $< --generate-hashes -o $@
endef

# Download dataset.
.PHONY: data
data:
	mkdir -p data
	-wget -nc -O data/images.zip https://vision.middlebury.edu/mview/data/data/temple.zip
	unzip data/images.zip -d data

.PHONY: requirements.txt
requirements.txt: requirements.in
	${compile-requirements}

.PHONY: requirements_ci.txt
requirements_ci.txt: requirements_ci.in
	${compile-requirements}

.PHONY: requirements
requirements: requirements.txt requirements_ci.txt
