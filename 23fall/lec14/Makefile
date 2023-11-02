lec  = lec14

all: $(lec).pdf

$(lec).pdf: $(lec).tex exp/Boxplot_vs_PDF.png exp/brownian-0.png exp/Multivariate_Gaussian.png exp/White_noise.png
	pdflatex $(lec)
	pdflatex $(lec)

exp/Boxplot_vs_PDF.png:
	mkdir -p exp
	wget -O exp/Boxplot_vs_PDF.png https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Boxplot_vs_PDF.svg/704px-Boxplot_vs_PDF.svg.png

exp/brownian-0.png:
	mkdir -p exp
	wget -O exp/brownian.gif https://upload.wikimedia.org/wikipedia/commons/5/51/Brownianmotion5particles150frame.gif
	convert -coalesce exp/brownian.gif exp/brownian-%d.png

exp/Multivariate_Gaussian.png:
	mkdir -p exp
	wget -O exp/Multivariate_Gaussian.png https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Multivariate_Gaussian.png/1024px-Multivariate_Gaussian.png

exp/White_noise.png:
	mkdir -p exp
	wget -O exp/White_noise.png https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/White_noise.svg/1024px-White_noise.svg.png

clean:
	-rm exp/*
	-rmdir exp
	-rm *~
	-rm *.aux *.log *.nav *.out *.snm *.toc

