lec  = lec10

$(lec).pdf: $(lec).tex exp/face_detection.jpg exp/fraunhofer.jpg exp/haarlike.png exp/rectangle.png exp/integralimage.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/face_detection.jpg:
	mkdir -p exp
	wget -O exp/face_detection.jpg https://upload.wikimedia.org/wikipedia/commons/e/ef/Face_detection.jpg

exp/fraunhofer.jpg:
	mkdir -p exp
	wget -O exp/fraunhofer.jpg https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Fraunhofer_-_Face_Detection.jpg/640px-Fraunhofer_-_Face_Detection.jpg


exp/haarlike.png:
	mkdir -p exp
	wget -O exp/haarlike.png https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/VJ_featureTypes.svg/640px-VJ_featureTypes.svg.png

exp/rectangle.png:
	mkdir -p exp
	wget -O exp/rectangle.png https://upload.wikimedia.org/wikipedia/commons/e/ee/Prm_VJ_fig3_computeRectangleWithAlpha.png

exp/integralimage.png:
	mkdir -p exp
	wget -O exp/integralimage.png https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Integral_image_application_example.svg/300px-Integral_image_application_example.svg.png

clean:
	-rm $(figs)
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

