lec  = lec22

$(lec).pdf: $(lec).tex exp/block-I-primary.png exp/PCA_seagrass.png exp/Autoencoder_schema.png exp/Autoencoder_structure.png exp/K_Means_Step_1.png exp/K_Means_Step_2.png exp/K_Means_Step_3.png exp/K_Means_Step_4.png exp/Insect_on_a_torus-0.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/block-I-primary.png:
	mkdir -p exp
	wget -O exp/block-I-primary.png https://marketing.illinois.edu/wp-content/uploads/2021/09/block-I-primary.png


exp/PCA_seagrass.png:
	mkdir -p exp
	wget -O ex/PCA_seagrass.png https://upload.wikimedia.org/wikipedia/commons/1/19/Principal_Component_Analyses_for_the_morphological_and_molecular_surveys_of_seagrass_meadows.png

exp/Autoencoder_schema.png:
	mkdir -p exp
	wget -O exp/Autoencoder_schema.png https://upload.wikimedia.org/wikipedia/commons/3/37/Autoencoder_schema.png

exp/Autoencoder_structure.png:
	mkdir -p exp
	wget -O exp/Autoencoder_structure.png https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png

exp/K_Means_Step_1.png:
	mkdir -p exp
	wget -O exp/K_Means_Step_1.png https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/K_Means_Example_Step_1.svg/498px-K_Means_Example_Step_1.svg.png

exp/K_Means_Step_2.png:
	mkdir -p exp
	wget -O exp/K_Means_Step_2.png https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/K_Means_Example_Step_2.svg/556px-K_Means_Example_Step_2.svg.png

exp/K_Means_Step_3.png:
	mkdir -p exp
	wget -O exp/K_Means_Step_3.png https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/K_Means_Example_Step_3.svg/556px-K_Means_Example_Step_3.svg.png

exp/K_Means_Step_4.png:
	mkdir -p exp
	wget -O exp/K_Means_Step_4.png https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/K_Means_Example_Step_4.svg/556px-K_Means_Example_Step_4.svg.png

exp/Insect_on_a_torus-0.png:
	wget -O exp/Insect_on_a_torus.gif https://upload.wikimedia.org/wikipedia/commons/6/6a/Insect_on_a_torus_tracing_out_a_non-trivial_geodesic.gif
	convert -coalesce exp/Insect_on_a_torus.gif exp/Insect_on_a_torus-%d.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

