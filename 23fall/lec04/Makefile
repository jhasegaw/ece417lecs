lec=lec04

$(lec).pdf: $(lec).tex exp/block-I-primary.png exp/dirichlet_in_decibels.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/block-I-primary.png:
	mkdir -p exp
	wget -O exp/block-I-primary.png https://marketing.illinois.edu/wp-content/uploads/2021/09/block-I-primary.png

exp/dirichlet_in_decibels.png:
	mkdir -p exp
	python makefigs.py

clean:
	-rm -rf exp
	-rm *~ *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz}
	-rm $(lec).pdf
