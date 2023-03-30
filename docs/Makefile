# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD = sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build

ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS)

ALLSPHINXAUTOOPTS = $(ALLSPHINXOPTS) \
	--ignore "*/$(BUILDDIR)/*" \
	--ignore "*/generated/*" \
	--ignore "*/jupyter_execute/*" \
	--ignore "*/.ipynb_checkpoints/*"


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)


allclean: clean
	rm -rf $(BUILDDIR)/*
	rm -rf generated/*
	rm -rf reference/generated/*

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(ALLSPHINXOPTS) $(O)


.PHONY: livehtml
livehtml:
	$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(ALLSPHINXAUTOOPTS) $(O)

# livehtml:
# 	$(SPHINXAUTOBUILD) $(ALLSPHINXAUTOOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"


# livehtml2:
# 	sphinx-autobuild 	\
# 		--ignore '_build/*' \
# 		--ignore 'reference/generated/*' \
# 		--ignore 'jupyter_execute/*' \
# 		--ignore 'notebooks/.ipynb_checkpoints/*' \
# 		. _build/html
