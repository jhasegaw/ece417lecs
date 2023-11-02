lec  = lec18
figs = exp/Plot-exponential-decay.png exp/fig0.png exp/fig5.png exp/1024px-LSTM_Cell.svg.png

all: $(lec).pdf 

$(lec).pdf: $(lec).tex $(figs)
	pdflatex $(lec)
	pdflatex $(lec)

$(figs):
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/7/7a/Plot-exponential-decay.png
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/LSTM_Cell.svg/1024px-LSTM_Cell.svg.png
	python makefigs.py

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

