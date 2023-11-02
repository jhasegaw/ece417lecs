lec  = lec15
downloaded = exp/Megawati_Sukarnoputri_0002.jpg exp/Colin_Powell_0001.jpg exp/Gloria_Macapagal_Arroyo_0001.jpg exp/Tony_Blair_0001.jpg

all: pdf

pdf: $(lec).tex $(downloaded) exp/gaussian_copula.png exp/gaussian_pdfs.png exp/gaussian_vectors.png exp/dependent_samples.png exp/sample_covariance.png exp/GaussianScatterPCA.png exp/orthogonal_projection.png exp/rotation-0.png
	pdflatex $(lec)
	pdflatex $(lec)

$(downloaded):
	mkdir -p exp
	wget -P exp http://vis-www.cs.umass.edu/lfw/images/Tony_Blair/Tony_Blair_0001.jpg
	wget -P exp http://vis-www.cs.umass.edu/lfw/images/Gloria_Macapagal_Arroyo/Gloria_Macapagal_Arroyo_0001.jpg
	wget -P exp http://vis-www.cs.umass.edu/lfw/images/Colin_Powell/Colin_Powell_0001.jpg
	wget -P exp http://vis-www.cs.umass.edu/lfw/images/Megawati_Sukarnoputri/Megawati_Sukarnoputri_0001.jpg
	wget -P exp http://vis-www.cs.umass.edu/lfw/images/Megawati_Sukarnoputri/Megawati_Sukarnoputri_0002.jpg

exp/gaussian_pdfs.png:
	mkdir -p exp
	wget -O exp/gaussian_pdfs.png https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/720px-Normal_Distribution_PDF.svg.png

exp/gaussian_vectors.png:
	mkdir -p exp
	wget -O exp/gaussian_vectors.png https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/EM-Gaussian-data.svg/434px-EM-Gaussian-data.svg.png

exp/dependent_samples.png:
	mkdir -p exp
	wget -O exp/dependent_samples.png https://upload.wikimedia.org/wikipedia/commons/f/f5/Multinormal_3_true.png

exp/gaussian_copula.png:
	mkdir -p exp
	wget -O exp/gaussian_copula.png https://upload.wikimedia.org/wikipedia/commons/6/6b/Gaussian_copula_gaussian_marginals.png

exp/sample_covariance.png:
	mkdir -p exp
	wget -O exp/sample_covariance.png https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/PCA_Pires.jpg/1280px-PCA_Pires.jpg

exp/GaussianScatterPCA.png:
	mkdir -p exp
	wget -O exp/GaussianScatterPCA.png https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/GaussianScatterPCA.svg/1024px-GaussianScatterPCA.svg.png

exp/orthogonal_projection.png:
	mkdir -p exp
	wget -O exp/orthogonal_projection.png https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Orthogonal_Projection_qtl1.svg/1024px-Orthogonal_Projection_qtl1.svg.png

exp/rotation-0.png:
	mkdir -p exp
	wget -O exp/rotation.gif https://upload.wikimedia.org/wikipedia/commons/4/4e/Diagonalization_as_rotation.gif
	convert -coalesce exp/rotation.gif exp/rotation-%d.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

