#!/usr/bin/env -S gnuplot -c
set term pngcairo size 1600, 1200
set output ARG2
set title ARG3
set xlabel "Experts"
set ylabel "Languages"
unset xtics
set ytics font ", 8"
plot ARG1 matrix rowheaders columnheaders using 1:2:3 with image
