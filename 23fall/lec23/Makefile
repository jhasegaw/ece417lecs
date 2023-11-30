lec  = lec23

$(lec).pdf: $(lec).tex exp/block-I-primary.png exp/1024px-LSTM_Cell.svg.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/block-I-primary.png:
	mkdir -p exp
	wget -O exp/block-I-primary.png https://marketing.illinois.edu/wp-content/uploads/2021/09/block-I-primary.png

exp/1024px-LSTM_Cell.svg.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/1024px-LSTM_Cell.svg.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

