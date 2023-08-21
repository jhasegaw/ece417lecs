lec=lec02

$(lec).pdf: $(lec).tex exp/mp1_xrmb_movie-0.png exp/mp1_result-0.png exp/mp1_procrustes_movie-0.png exp/mp1_procrustes_triangles-0.png exp/Checkerboard_identity.png exp/Checkerboard_reflection.png exp/Checkerboard_scale.png exp/Checkerboard_rotate.png exp/Checkerboard_shear.png exp/block-I-primary.png exp/SVD_visualization.png exp/Ortho_projection.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/block-I-primary.png:
	mkdir -p exp
	wget -O exp/block-I-primary.png https://marketing.illinois.edu/wp-content/uploads/2021/09/block-I-primary.png

exp/mp1_xrmb_movie-0.png:
	mkdir -p exp
	convert -coalesce mp1_xrmb_movie.mp4 exp/mp1_xrmb_movie-%d.png

exp/mp1_result-0.png:
	mkdir -p exp
	convert -coalesce mp1_result.mp4 exp/mp1_result-%d.png

exp/mp1_procrustes_movie-0.png:
	mkdir -p exp
	convert -coalesce mp1_procrustes_movie.mp4 exp/mp1_procrustes_movie-%d.png

exp/mp1_procrustes_triangles-0.png:
	mkdir -p exp
	convert -coalesce mp1_procrustes_triangles.mp4 exp/mp1_procrustes_triangles-%d.png

exp/Checkerboard_identity.png:
	mkdir -p exp
	wget -O exp/Checkerboard_identity.png https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Checkerboard_identity.svg/768px-Checkerboard_identity.svg.png

exp/Checkerboard_reflection.png:
	mkdir -p exp
	wget -O exp/Checkerboard_reflection.png https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Checkerboard_reflection.svg/768px-Checkerboard_reflection.svg.png

exp/Checkerboard_scale.png:
	mkdir -p exp
	wget -O exp/Checkerboard_scale.png https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Checkerboard_scale.svg/640px-Checkerboard_scale.svg.png

exp/Checkerboard_rotate.png:
	mkdir -p exp
	wget -O exp/Checkerboard_rotate.png https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Checkerboard_rotate.svg/768px-Checkerboard_rotate.svg.png

exp/Checkerboard_shear.png:
	mkdir -p exp
	wget -O exp/Checkerboard_shear.png https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Checkerboard_shear.svg/768px-Checkerboard_shear.svg.png

exp/SVD_visualization.png:
	mkdir -p exp
	wget -O exp/SVD_visualization.png https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Singular_value_decomposition_visualisation.svg/659px-Singular_value_decomposition_visualisation.svg.png

exp/Ortho_projection.png:
	mkdir -p exp
	wget -O exp/Ortho_projection.png https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Ortho_projection.svg/1280px-Ortho_projection.svg.png
clean:
	-rm -rf exp
	-rm *~ *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz}
	-rm $(lec).pdf
