#! /bin/sh

find . -name "*.jmd" | while read f
do
  dir=$(dirname $f)
  f=$(basename $f)
  olddir=$PWD
  cd $dir
  julia --project=$olddir -e "using JSOTutorials; JSOTutorials.conversions(\"$f\")"
  cd $olddir
done
