lec  = lec26
figs = exp/1024px-LSTM_Cell.svg.png

all: $(lec).pdf 

$(lec).pdf: $(lec).tex $(figs)
	pdflatex $(lec)
	pdflatex $(lec)

$(figs):
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/1024px-LSTM_Cell.svg.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

