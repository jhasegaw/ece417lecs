lec=lec07

$(lec).pdf: $(lec).tex exp/block-I-primary.png exp/Arc_length-0.png exp/Tangent.png exp/Directional.png exp/Pushforward.png exp/Ray.png exp/linear_regression.png exp/circle_tangent-0.png exp/rotation_matrix.png
	pdflatex $(lec)
	pdflatex $(lec)
	echo

exp/block-I-primary.png:
	mkdir -p exp
	wget -O exp/block-I-primary.png https://marketing.illinois.edu/wp-content/uploads/2021/09/block-I-primary.png


exp/Arc_length-0.png:
	mkdir -p exp
	wget -O exp/Arc_length.gif https://upload.wikimedia.org/wikipedia/commons/d/dc/Arc_length.gif
	convert exp/Arc_length.gif exp/Arc_length.png

exp/Tangent.png:
	mkdir -p exp
	wget -O exp/Tangent.png https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Tangent_to_a_curve.svg/1024px-Tangent_to_a_curve.svg.png

exp/Directional.png:
	mkdir -p exp
	wget -O exp/Directional.png https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Directional_derivative_contour_plot.svg/775px-Directional_derivative_contour_plot.svg.png

exp/Pushforward.png:
	mkdir -p exp
	wget -O exp/Pushforward.png https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Pushforward.svg/1024px-Pushforward.svg.png

exp/Ray.png:
	mkdir -p exp
	wget -O exp/Ray.png https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Ray_%28A%2C_B%2C_C%29.svg/1024px-Ray_%28A%2C_B%2C_C%29.svg.png

exp/circle_tangent-0.png:
	mkdir -p exp
	wget -O exp/circle_tangent.gif https://upload.wikimedia.org/wikipedia/commons/7/7d/Circle%2B3vectors_animated.gif
	convert exp/circle_tangent.gif exp/circle_tangent.png

exp/linear_regression.png:
	mkdir -p exp
	wget -O exp/linear_regression.png https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Linear_least_squares_example2.svg/782px-Linear_least_squares_example2.svg.png


exp/rotation_matrix.png:
	mkdir -p exp
	wget -O exp/rotation_matrix.png https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Visual_Derivation_of_Equations_For_Rotation_In_2D.svg/921px-Visual_Derivation_of_Equations_For_Rotation_In_2D.svg.png

clean:
	-rm -rf exp
	-rm *~ *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz}
	-rm $(lec).pdf
