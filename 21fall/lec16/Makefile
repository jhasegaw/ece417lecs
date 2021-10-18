lec  = lec16
figs = exp/An_example_of_HMM.png exp/Tychonoff.jpg

all: $(lec).pdf

$(lec).pdf: $(lec).tex $(figs)
	pdflatex $(lec)
	pdflatex $(lec)

$(figs):
	mkdir -p exp
	wget -P  exp https://upload.wikimedia.org/wikipedia/commons/0/0c/An_example_of_HMM.png
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/b/bd/Tychonoff.jpg

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

