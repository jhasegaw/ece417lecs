lec  = lec12

all: $(lec).pdf

$(lec).pdf: $(lec).tex exp/Gumball_machines_Dallas_2008.jpg exp/seatedlibertydollar.jpg exp/640px-Convolutional_code_trellis_diagram.svg.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/Gumball_machines_Dallas_2008.jpg:
	mkdir -p exp
	wget -O exp/Gumball_machines_Dallas_2008.jpg https://upload.wikimedia.org/wikipedia/commons/1/1f/Gumball_machines_Dallas_2008.jpg

exp/seatedlibertydollar.jpg:
	mkdir -p exp
	wget -O exp/seatedlibertydollar.jpg https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/1859-O_%241.jpg/640px-1859-O_%241.jpg

exp/640px-Convolutional_code_trellis_diagram.svg.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Convolutional_code_trellis_diagram.svg/640px-Convolutional_code_trellis_diagram.svg.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

