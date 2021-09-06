target = lec02
byproducts = *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz} 

all:	figs pdf

pdf:
	pdflatex $(target)
	pdflatex $(target)

figs:
	python makefigs.py

clean:
	-rm $(byproducts)
	-rmdir exp

superclean:
	-rm $(byproducts) $(target).pdf
	-rmdir exp
