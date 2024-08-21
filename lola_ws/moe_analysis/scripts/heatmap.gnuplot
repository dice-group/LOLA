#!/usr/bin/env -S gnuplot -c
set term pngcairo size 1600, 1200
set output ARG2
set title ARG3
set xlabel "Experts"
set ylabel "Languages"
unset xtics
set ytics font ", 8"

layers = 12
experts = 16

set cbtics ("0" 0, "1/".experts 1.0/experts, "2/".experts 2*1.0/experts, "1/2" 0.5)

set palette defined (0 "blue", 1.0/experts "white", 2*1.0/experts "yellow", 0.6 "red")

set for [layer=0:layers-2] arrow from experts - 0.5 + experts * layer, -0.5 to experts - 0.5 + experts * layer, 105.5 nohead front

plot ARG1 matrix rowheaders columnheaders using 1:2:3 with image
