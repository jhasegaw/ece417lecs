lec  = lec17

$(lec).pdf: $(lec).tex exp/Conv_layer.png exp/800px-Recurrent_neural_network_unfold.svg.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/Conv_layer.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/6/68/Conv_layer.png

exp/800px-Recurrent_neural_network_unfold.svg.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/800px-Recurrent_neural_network_unfold.svg.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

