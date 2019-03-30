#!/bin/bash

if [ -n "$1" ]
then
number=$1
else
number=10
fi

for (( i=1; i <= $number; i++ ))
do
../Source/main

percent=$(printf %.0f\\n "$((10**2 * i/number))")
echo -ne "\rProgress: $percent%"
done

echo

python calculate_average_times.py

cat ../Results/txt/average_times.txt
