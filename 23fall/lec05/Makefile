lec=lec05

$(lec).pdf: $(lec).tex exp/block-I-primary.png exp/Hamming_spectrum.png exp/Multidimensional_Analysis_Filter_Banks.jpg exp/Multidimensional_Synthesis_Filter_Banks.jpg exp/Multidimensional_M_channel_Filter_Banks.jpg exp/Spectrogram-19thC.png exp/hamming_window.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/hamming_window.png:
	mkdir -p exp
	python makefigs.py

exp/block-I-primary.png:
	mkdir -p exp
	wget -O exp/block-I-primary.png https://marketing.illinois.edu/wp-content/uploads/2021/09/block-I-primary.png

exp/dirichlet_in_decibels.png:
	mkdir -p exp
	python makefigs.py

exp/Hamming_spectrum.png:
	mkdir -p exp
	wget -O exp/Hamming_spectrum.png https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Window_function_and_frequency_response_-_Hamming_%28alpha_%3D_0.53836%2C_n_%3D_0...N%29.svg/640px-Window_function_and_frequency_response_-_Hamming_%28alpha_%3D_0.53836%2C_n_%3D_0...N%29.svg.png

exp/Multidimensional_Analysis_Filter_Banks.jpg:
	mkdir -p exp
	wget -O exp/Multidimensional_Analysis_Filter_Banks.jpg https://upload.wikimedia.org/wikipedia/en/d/d4/Multidimensional_Analysis_Filter_Banks.jpg

exp/Multidimensional_Synthesis_Filter_Banks.jpg:
	mkdir -p exp
	wget -O exp/Multidimensional_Synthesis_Filter_Banks.jpg https://upload.wikimedia.org/wikipedia/en/6/6c/Multidimensional_Synthesis_Filter_Banks.jpg

exp/Multidimensional_M_channel_Filter_Banks.jpg:
	mkdir -p exp
	wget -O exp/Multidimensional_M_channel_Filter_Banks.jpg https://upload.wikimedia.org/wikipedia/en/d/df/Multidimensional_M_channel_Filter_Banks.jpg

exp/Spectrogram-19thC.png:
	mkdir -p exp
	wget -O exp/Spectrogram-19thC.png https://upload.wikimedia.org/wikipedia/commons/c/c5/Spectrogram-19thC.png


clean:
	-rm -rf exp
	-rm *~ *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz}
	-rm $(lec).pdf
