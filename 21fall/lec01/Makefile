all:	figs pdf

pdf:
	pdflatex lec01
	pdflatex lec01

figs:
	python makefigs.py

clean:
	-rm *~ exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz}
	-rmdir exp

superclean:
	-rm *~ lec01.pdf exp/* *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz}
	-rmdir exp
