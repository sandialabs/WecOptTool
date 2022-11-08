#  makefile for Sphinx documentation

SPHINXOPTS    = -W
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
EXAMPLEDIR    = source/_examples
APIDIR        = source/api_docs

# Put it first so that "make" is equivalent to "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

clean:
	@-rm -rf $(BUILDDIR) $(EXAMPLEDIR) $(APIDIR)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	python3 $(SOURCEDIR)/make_theory_animations.py
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@sed '/<section id="package">/,/<\/section>/d' _build/html/index.html > tmp; mv tmp _build/html/index.html