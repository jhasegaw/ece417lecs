lec=lec06

$(lec).pdf: $(lec).tex exp/block-I-primary.png exp/Spectrogram-19thC.png exp/unknown_phase.png exp/twoconstraints_iterate.png exp/cosine_stft.png exp/cosine_mstft.png exp/cosine_rstft.png exp/cosine_ola.png exp/cosine_stft2.png exp/cosine_mstft2.png exp/cosine_iterate1.png exp/cosine_iterate2.png exp/cosine_iterate3.png exp/cosine_iterate4.png
	pdflatex $(lec)
	pdflatex $(lec)
	echo

exp/block-I-primary.png:
	mkdir -p exp
	wget -O exp/block-I-primary.png https://marketing.illinois.edu/wp-content/uploads/2021/09/block-I-primary.png


exp/Spectrogram-19thC.png:
	mkdir -p exp
	wget -O exp/Spectrogram-19thC.png https://upload.wikimedia.org/wikipedia/commons/c/c5/Spectrogram-19thC.png

exp/unknown_phase.png:
	python makefigs.py unknown_phase

exp/twoconstraints_iterate.png:
	python makefigs.py twoconstraints

exp/cosine_stft.png:
	python makefigs.py cosine_stft

exp/cosine_mstft.png:
	python makefigs.py cosine_mstft

exp/cosine_rstft.png:
	python makefigs.py cosine_rstft

exp/cosine_ola.png:
	python makefigs.py cosine_ola

exp/cosine_stft2.png:
	python makefigs.py cosine_stft2

exp/cosine_mstft2.png:
	python makefigs.py cosine_mstft2

exp/cosine_iterate1.png:
	python makefigs.py cosine_iterate1

exp/cosine_iterate2.png:
	python makefigs.py cosine_iterate2

exp/cosine_iterate3.png:
	python makefigs.py cosine_iterate3

exp/cosine_iterate4.png:
	python makefigs.py cosine_iterate4

clean:
	-rm -rf exp
	-rm *~ *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz}
	-rm $(lec).pdf
