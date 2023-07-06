#!/bin/bash

cd $HOME/delft/doc/

for f in *.drawio; do
    drawio -x -f png --scale 2.0 --border 20 \
        -o img/${f%.*}.png $f
done

cd $HOME/delft/


