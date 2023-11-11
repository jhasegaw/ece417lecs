lec  = lec21

$(lec).pdf: $(lec).tex exp/block-I-primary.png exp/Neural_Abstraction_Pyramid.jpg exp/PCA_vs_Linear_Autoencoder.png exp/Autoencoder_schema.png exp/Cauchy-Schwarz_inequation_in_Euclidean_plane.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/block-I-primary.png:
	mkdir -p exp
	wget -O exp/block-I-primary.png https://marketing.illinois.edu/wp-content/uploads/2021/09/block-I-primary.png

exp/Cauchy-Schwarz_inequation_in_Euclidean_plane.png:
	mkdir -p exp
	wget -O exp/foo.gif https://upload.wikimedia.org/wikipedia/commons/2/27/Cauchy-Schwarz_inequation_in_Euclidean_plane.gif
	convert exp/foo.gif exp/Cauchy-Schwarz_inequation_in_Euclidean_plane.png

exp/Autoencoder_schema.png:
	mkdir -p exp
	wget -O exp/Autoencoder_schema.png https://upload.wikimedia.org/wikipedia/commons/3/37/Autoencoder_schema.png

exp/PCA_vs_Linear_Autoencoder.png:
	mkdir -p exp
	wget -O exp/PCA_vs_Linear_Autoencoder.png https://upload.wikimedia.org/wikipedia/commons/0/0b/PCA_vs_Linear_Autoencoder.png


exp/Neural_Abstraction_Pyramid.jpg:
	mkdir -p exp
	wget -O exp/Neural_Abstraction_Pyramid.jpg https://upload.wikimedia.org/wikipedia/commons/d/d9/Neural_Abstraction_Pyramid.jpg

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

