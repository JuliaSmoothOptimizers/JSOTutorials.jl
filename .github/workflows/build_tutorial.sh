#!/bin/bash

FILE=$1
if [ ! -f "$FILE" ];
then
  echo "ERROR: $FILE is not a file"
  return
fi

folder=$(dirname "$FILE")
file=$(basename "$FILE")

folder_without_tutorial_prefix=$(echo $folder | cut -d'/' -f2-)
file_change_suffix=$(basename -s .jmd $file).md
MD_FILE=markdown/$folder_without_tutorial_prefix/$file_change_suffix
OUTPUT_FILE=parsed/$folder_without_tutorial_prefix/$file_change_suffix
mkdir -p $(dirname $OUTPUT_FILE)

echo ">> Weaving $file in $folder"
julia --project -e """
using Pkg;
Pkg.instantiate();
using JSOTutorials;
folder = split(\"$folder\", \"/\")[2:end] # to remove tutorials prefix;
folder = join(folder, \"/\");
JSOTutorials.weave_file(folder, \"$file\")
JSOTutorials.parse_markdown_into_franklin(\"$MD_FILE\", \"$OUTPUT_FILE\")
"""
