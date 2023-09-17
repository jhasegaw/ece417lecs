lec  = lec08

all: $(lec).pdf

$(lec).pdf: $(lec).tex exp/nntrain0.png exp/voronoi.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/nntrain0.png:
	mkdir -p exp
	python makefigs.py

exp/voronoi.png:
	mkdir -p exp
	wget -O exp/voronoi.png https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Euclidean_Voronoi_diagram.svg/768px-Euclidean_Voronoi_diagram.svg.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

