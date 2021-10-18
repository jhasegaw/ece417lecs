lec  = lec12
downloaded = exp/Megawati_Sukarnoputri_0002.jpg exp/Colin_Powell_0001.jpg exp/Gloria_Macapagal_Arroyo_0001.jpg exp/Tony_Blair_0001.jpg

all: pdf

pdf: $(lec).tex $(downloaded)
	pdflatex $(lec)
	pdflatex $(lec)

$(downloaded):
	mkdir -p exp
	wget -P exp http://vis-www.cs.umass.edu/lfw/images/Tony_Blair/Tony_Blair_0001.jpg
	wget -P exp http://vis-www.cs.umass.edu/lfw/images/Gloria_Macapagal_Arroyo/Gloria_Macapagal_Arroyo_0001.jpg
	wget -P exp http://vis-www.cs.umass.edu/lfw/images/Colin_Powell/Colin_Powell_0001.jpg
	wget -P exp http://vis-www.cs.umass.edu/lfw/images/Megawati_Sukarnoputri/Megawati_Sukarnoputri_0001.jpg
	wget -P exp http://vis-www.cs.umass.edu/lfw/images/Megawati_Sukarnoputri/Megawati_Sukarnoputri_0002.jpg


clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

