#!/bin/bash

mkdir -p exp
cd exp
wget https://upload.wikimedia.org/wikipedia/commons/0/0c/An_example_of_HMM.png
wget https://upload.wikimedia.org/wikipedia/commons/7/73/Viterbi_animated_demo.gif
convert -coalesce Viterbi_animated_demo.gif Viterbi_animated_demo.png
cd ..
