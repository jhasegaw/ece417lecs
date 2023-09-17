lec  = lec09

$(lec).pdf: $(lec).tex exp/gradient_descent.png exp/800px-Typical_cnn.png exp/backward0.png exp/Paracas.jpg exp/450px-Sky-3.jpg exp/Max_pooling.png exp/convolution_correlation.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/800px-Typical_cnn.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/800px-Typical_cnn.png


exp/Paracas.jpg:
	mkdir -p exp
	wget -O exp/Paracas.jpg 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Paracas_National_Reserve%2C_Ica%2C_Peru-3April2011.jpg/800px-Paracas_National_Reserve%2C_Ica%2C_Peru-3April2011.jpg'

exp/450px-Sky-3.jpg:
	mkdir -p exp
	wget -O exp/450px-Sky-3.jpg https://upload.wikimedia.org/wikipedia/commons/4/4b/Cumulus_clouds_seen_from_10%2C000_meters_above_the_ground%2C_2010.jpg

exp/Max_pooling.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png

exp/gradient_descent.png:
	mkdir -p exp
	python makefigs.py

exp/convolution_correlation.png:
	mkdir -p exp
	wget -O exp/convolution_correlation.png https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Comparison_convolution_correlation.svg/1024px-Comparison_convolution_correlation.svg.png

clean:
	-rm $(figs)
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

