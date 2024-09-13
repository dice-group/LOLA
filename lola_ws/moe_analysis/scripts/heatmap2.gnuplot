#!/usr/bin/env -S gnuplot -c
set term pngcairo size 1600, 1220 font 'Times,35'
# set terminal pdfcairo size 16,12 enhanced font 'Times,10'
set output ARG2 . '.png'
#set title ARG3
set xlabel "Experts" font 'Times,48'
set ylabel "Languages" font 'Times,48'
unset xtics
#set ytics font ", 8"
unset ytics

layers = 12
experts = 16

set cbrange [0:1]
set cbtics ("0" 0, "1/".experts 1.0/experts, "2/".experts 2*1.0/experts, 4*1.0/experts, "1/2" 0.5)

# set palette defined (0 "blue", 1.0/experts "white", 2*1.0/experts "yellow", 0.75 "red", 1.0 "red")
set palette defined (0 "white", 1.0/experts "light-green", 2*1.0/experts "yellow", 4*1.0/experts "light-red", 0.75 "red", 1.0 "red")

set for [layer=0:layers-2] arrow from experts - 0.5 + experts * layer, -0.5 to experts - 0.5 + experts * layer, 105.5 nohead front

plot ARG1 matrix rowheaders columnheaders using 1:2:3 with image
