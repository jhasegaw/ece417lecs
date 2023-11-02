lec  = lec19

$(lec).pdf: $(lec).tex exp/melspectrogram.png exp/haplogroup.png exp/lstm.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/melfilters.png:
	python makefigs.py

exp/melspectrogram.png:
	mkdir -p exp
	wget -O exp/melspectrogram.png https://librosa.org/doc/main/_images/librosa-feature-melspectrogram-1.png

exp/haplogroup.png:
	mkdir -p exp
	wget -O exp/haplogroup.png https://upload.wikimedia.org/wikipedia/commons/6/69/PCA_of_Haplogroup_J_using_37_STRs.png

exp/lstm.png:
	mkdir -p exp
	wget -O exp/lstm.png https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Long_Short-Term_Memory.svg/1280px-Long_Short-Term_Memory.svg.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

