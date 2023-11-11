lec  = lec20

$(lec).pdf: $(lec).tex exp/block-I-primary.png exp/autoencoder2.png exp/intersectingplanes.png exp/changeofbasis.png exp/sparseautoencoder.png exp/kleinbottle.png exp/deepautoencoder.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/block-I-primary.png:
	mkdir -p exp
	wget -O exp/block-I-primary.png https://marketing.illinois.edu/wp-content/uploads/2021/09/block-I-primary.png

exp/autoencoder2.png:
	mkdir -p exp
	wget -O exp/autoencoder2.png https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Autoencoder_schema.png/659px-Autoencoder_schema.png

exp/changeofbasis.png:
	mkdir -p exp
	wget -O exp/changeofbasis.png https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Change_of_basis_22.svg/780px-Change_of_basis_22.svg.png

exp/pca.png:
	mkdir -p exp
	wget -O exp/pca.png https://???

exp/intersectingplanes.png:
	mkdir -p exp
	wget -O exp/intersectingplanes.png https://upload.wikimedia.org/wikipedia/commons/7/7b/PlaneIntersection.png

exp/sparseautoencoder.png:
	mkdir -p exp
	wget -O exp/sparseautoencoder.png https://upload.wikimedia.org/wikipedia/commons/8/83/Autoencoder_sparso.png

exp/kleinbottle.png:
	mkdir -p exp
	wget -O exp/kleinbottle.png https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Klein_bottle.svg/533px-Klein_bottle.svg.png

exp/deepautoencoder.png:
	mkdir -p exp
	wget -O exp/deepautoencoder.png https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

