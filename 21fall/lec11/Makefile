all: pdf

pdf: lec11.tex exp/classlabels.png exp/1024px-Perceptron_example.svg.png
	pdflatex lec11
	pdflatex lec11


exp/classlabels.png:
	mkdir -p exp
	python makefigs.py

exp/Perceptron_example.svg/1024px-Perceptron_example.svg.png:
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Perceptron_example.svg/1024px-Perceptron_example.svg.png
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/450px-Kosaciec_szczecinkowaty_Iris_setosa.jpg
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/800px-Iris_versicolor_3.jpg
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/736px-Iris_virginica.jpg
	wget -P exp https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Iris_dataset_scatterplot.svg/600px-Iris_dataset_scatterplot.svg.png




clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc
