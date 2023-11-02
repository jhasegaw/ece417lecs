lec  = lec13

all: $(lec).pdf

$(lec).pdf: $(lec).tex exp/lagrange.png exp/An_example_of_HMM.png exp/Viterbi_animated_demo-0.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/lagrange.png:
	mkdir -p exp
	wget -O exp/lagrange.png https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/LagrangeMultipliers2D.svg/640px-LagrangeMultipliers2D.svg.png

exp/An_example_of_HMM.png:
	mkdir -p exp
	wget -P  exp https://upload.wikimedia.org/wikipedia/commons/0/0c/An_example_of_HMM.png


exp/Viterbi_animated_demo-0.png:
	mkdir -p exp
	wget -O exp/Viterbi_animated_demo.gif https://upload.wikimedia.org/wikipedia/commons/7/73/Viterbi_animated_demo.gif
	convert -coalesce exp/Viterbi_animated_demo.gif exp/Viterbi_animated_demo-%d.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

