# JSO Tutorials

Curated repository of tutorials regarding all things JSO.

## Using

Check the specific folder.

*TODO: table of contents*

## Contributing

We use `Weave.jl` to write `.jmd` files, then run `JSOTutorials.conversions(filename)` to generate the specific files.

### Guidelines:

- Follow the folder structure `subject`/`tutorial-name/`tutorial-name.ext`;
- File and folders name should be `lower-case-separated-by-dashes`;
- `.jmd` is the main file, but all files have to be generated;
- Work on environments (`pkg> activate .`) and save the `Manifest.toml`.

## TODO

- Table of contents
- Test on travis that the files are generated correctly
- (?) Generate files directly on travis and push them
