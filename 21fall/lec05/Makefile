all: pdf

pdf: lec05.tex exp/delta.png
	pdflatex lec05
	pdflatex lec05


exp/delta.png:
	mkdir -p exp
	cd exp
	wget https://upload.wikimedia.org/wikipedia/commons/a/aa/Fourier2.jpg
	convert Fourier2.jpg -set colorspace Gray -separate -average -scale 200x200 fourier.jpg
	cd ..
	python makefigs.py

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc
