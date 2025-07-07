# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= --fail-on-warning
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

api_docs: ../pyccel/**/*.py
	@sphinx-apidoc -e -o source/api -H "Internal library reference" ../pyccel

html: api_docs
	@mkdir -p source/docs
	@cp *.md source/docs
	@cp -r tutorials source/tutorials
	@mkdir -p source/developer_docs
	@cp ../developer_docs/*.md source/developer_docs
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@rm -f source/*.md
	@rm -rf source/developer_docs
	@rm -rf source/api
	@rm -rf build

%:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
