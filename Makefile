define compile-requirements =
pip-compile $< --generate-hashes -o $@
endef

# Download dataset.
.PHONY: data
data:
	mkdir -p data
	-wget -nc -O data/kusvod2.tar.gz http://cmp.felk.cvut.cz/data/geometry2view/Lebeda-2012-kusvod2.tar.gz
	tar -xzf data/kusvod2.tar.gz -C data

.PHONY: requirements.txt
requirements.txt: requirements.in
	${compile-requirements}

.PHONY: requirements_ci.txt
requirements_ci.txt: requirements_ci.in
	${compile-requirements}

.PHONY: requirements
requirements: requirements.txt requirements_ci.txt
