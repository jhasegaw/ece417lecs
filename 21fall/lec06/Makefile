pdf: lec06.tex exp/Opticfloweg.png exp/Optical-flow-estimation-for-different-sequences.jpg exp/tendon/0001.png exp/512px-Gradient2.svg.png exp/Intensity_image_with_gradient_images.png exp/Block-matching_algorithm.png
	pdflatex lec06
	pdflatex lec06
	pdflatex lec06_novideo
	pdflatex lec06_novideo

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

exp/Opticfloweg.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/5/55/Opticfloweg.png

exp/Optical-flow-estimation-for-different-sequences.jpg:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/2/26/Optical-flow-estimation-for-different-sequences.jpg

exp/tendon/0001.png:
	mkdir -p exp
	cd exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/c/c0/Tendon-motion-tracking-in-an-ultrasound-image-sequence-using-optical-flow-based-block-matching-12938_2017_335_MOESM9_ESM.ogv
	mkdir -p exp/tendon
	ffmpeg -i exp/Tendon-motion-tracking-in-an-ultrasound-image-sequence-using-optical-flow-based-block-matching-12938_2017_335_MOESM9_ESM.ogv exp/tendon/%d.png

exp/512px-Gradient2.svg.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Gradient2.svg/512px-Gradient2.svg.png

exp/Intensity_image_with_gradient_images.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/6/67/Intensity_image_with_gradient_images.png

exp/Block-matching_algorithm.png:
	mkdir -p exp
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/1/15/Block-matching_algorithm.png
