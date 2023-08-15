lec=lec01

$(lec).pdf: $(lec).tex
	pdflatex $(lec)
	pdflatex $(lec)

clean:
	-rm -rf exp
	-rm *~ *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz}
	-rm $(lec).pdf
