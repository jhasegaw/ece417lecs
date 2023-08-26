lec  = lec20
figs = exp/gradient_descent.png exp/forward0.png exp/800px-Typical_cnn.png exp/backward0.png exp/800px-Paracas_National_Reserve,_Ica,_Peru-3April2011.jpg exp/450px-Sky-3.jpg exp/Max_pooling.png

all: $(lec).pdf 

$(lec).pdf: $(lec).tex $(figs)
	pdflatex $(lec)
	pdflatex $(lec)

$(figs):
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/800px-Typical_cnn.png
	wget -P exp 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Paracas_National_Reserve%2C_Ica%2C_Peru-3April2011.jpg/800px-Paracas_National_Reserve%2C_Ica%2C_Peru-3April2011.jpg'
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Sky-3.jpg/450px-Sky-3.jpg
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png
	python makefigs.py

clean:
	-rm $(figs)
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

