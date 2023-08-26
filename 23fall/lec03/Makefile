lec=lec03

$(lec).pdf: $(lec).tex exp/block-I-primary.png exp/PointSetTriangulations.png exp/Barycentric.png exp/original.png exp/mp1_result-0.png exp/mp1_procrustes_movie-0.png exp/mp1_procrustes_triangles-0.png
	pdflatex $(lec)
	pdflatex $(lec)


exp/block-I-primary.png:
	mkdir -p exp
	wget -O exp/block-I-primary.png https://marketing.illinois.edu/wp-content/uploads/2021/09/block-I-primary.png

exp/mp1_result-0.png:
	mkdir -p exp
	convert -coalesce mp1_result.mp4 exp/mp1_result-%d.png

exp/mp1_procrustes_movie-0.png:
	mkdir -p exp
	convert -coalesce mp1_procrustes_movie.mp4 exp/mp1_procrustes_movie-%d.png

exp/mp1_procrustes_triangles-0.png:
	mkdir -p exp
	convert -coalesce mp1_procrustes_triangles.mp4 exp/mp1_procrustes_triangles-%d.png

exp/PointSetTriangulations.png:
	mkdir -p exp
	wget -O exp/PointSetTriangulations.png https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/PointSetTriangulations.svg/400px-PointSetTriangulations.svg.png

exp/Barycentric.png:
	mkdir -p exp
	wget -O exp/Barycentric.png https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/TriangleBarycentricCoordinates.svg/684px-TriangleBarycentricCoordinates.svg.png

exp/original.png:
	mkdir -p exp
	python makefigs.py

clean:
	-rm -rf exp
	-rm *~ *.{aux,bbl,blg,dvi,log,out,toc,nav,snm,synctex.gz}
	-rm $(lec).pdf
