lec  = lec13
figs = exp/eminequality.png

all: $(lec).pdf

$(lec).pdf: $(lec).tex $(figs)
	pdflatex $(lec)
	pdflatex $(lec)

$(figs):
	mkdir -p exp
	python makefigs.py


clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

