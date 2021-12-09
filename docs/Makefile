#  makefile for Sphinx documentation

SPHINXOPTS    = -W -n
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
EXAMPLEDIR    = source/_examples

# Put it first so that "make" is equivalent to "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

clean:
	@-rm -rf $(BUILDDIR) $(EXAMPLEDIR)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
