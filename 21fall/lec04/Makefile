lec01.pdf: lec04.tex exp/Normal_Distribution.png exp/MultivariateNormal.png
	pdflatex lec04
	pdflatex lec04

exp/Normal_Distribution.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/1920px-Normal_Distribution_PDF.svg.png 
	mv exp/1920px-Normal_Distribution_PDF.svg.png  exp/Normal_Distribution.png

exp/MultivariateNormal.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/8/8e/MultivariateNormal.png

clean:
	rm -rf exp
	~/d/scr/cleanlatex
