lec  = lec14
figs = Gumball_machines_Dallas_2008.jpg exp/seatedlibertydollar.jpg exp/640px-Convolutional_code_trellis_diagram.svg.png

all: $(lec).pdf

$(lec).pdf: $(lec).tex $(figs)
	pdflatex $(lec)
	pdflatex $(lec)

$(figs):
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/1/1f/Gumball_machines_Dallas_2008.jpg
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/1859-O_%241.jpg/640px-1859-O_%241.jpg
	mv exp/640px-1859*.jpg exp/seatedlibertydollar.jpg
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Convolutional_code_trellis_diagram.svg/640px-Convolutional_code_trellis_diagram.svg.png
clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

