# JSO Tutorials

Curated repository of tutorials regarding all things JSO.
See the renderer tutorials in <https://jso.dev>.

## Development

All tutorials are in the `tutorials` folder, inside at least an additional folder.
In most cases there is a single `index.jmd`, although it should be possible to have multiple tutorials under the same folder.

The most important part of the folders is that they should have their own `Project.toml`, i.e., they are Julia environments.
So, to develop a new tutorial, create the folder, and open Julia on the folder (you can check that you are in the correct place using `pwd()`.)

### Creating the environment

Assuming you are in the correct folder with the Julia REPL open, press `]` to enter `pkg` mode, and enter `activate .` to activate the environment. Then, you can add the necessary packages for your tutorial.
After the packages are added, the `Project.toml` file should be created.

**Do not to add Weave!** See the building section below.

### Tutorial header

The tutorial `.jmd` file should have a header like the following:

```yaml
title: "Title of the tutorial"
tags: ["tag1", "tag2"]
author: "One Author and Other Author"
```

**Please add sufficient tags**.

### Building and previewing locally the tutorial

During the development of the tutorial, you will probably want to rerun an rebuild the tutorial several times.
To do that, you should

- **open Julia in the root folder of JSOTutorials.jl**,
- `pkg> activate .`
- `pkg> instantiate`
- `julia> using JSOTutorials`
- `JSOTutorials.weave_file(folder, file)`.

For reasons that are not entirely clear, `folder` should be the folder **without the tutorials/ prefix**.
For instance, I can call `JSOTutorials.weave_file("create-a-manual-model", "index.jmd")`.

This will generate a `markdown/create-a-manual-model/index.md` file, which you can preview, for instance, using VSCode.

### If you want to check the end result (optional)

The parsed file (contained in `markdown/`) is not exactly what will be sent to the website, because of the Franklin format expected.
To fix it, we actually run a separate script.

**You don't have to do this step.**

The final version will be generated when your code is accepted in the `main` branch, and this is automatic.

That being said, if are doing some fixes or improvements on this step - or are just curious - then you can do so by running in the shell, the following command:

```shell
bash .github/workflows/build_tutorial.sh tutorials/create-a-manual-model/index.jmd
```

This will run the same `weave_file` command that we used before, and the function `JSOTutorials.parse_markdown_into_franklin`, creating a file inside the folder `parsed/`.

This function does a few magic things:

- It parses the YAML header and transforms into a Franklin header.
- It changes output fenced code to type `plaintext`.

And it can possibly do more in the future.

### Creating your Pull Request

If you haven't create a fork, do so now.

If you cloned your repository, then `origin` will refer to your fork, so you can use `origin` instead of `myfork` below.

If you cloned `JSOTutorials` from JSO, then you have to add your repository and a git remote.
Do so with the following command:

```shell
git remote add myfork https://github.com/YOURUSER/JSOTutorials.jl
```

Now, create a branch, commit, and push.

```shell
git checkout -b some-tutorial
git add tutorial/some-tutorial
git commit -am "New tutorial: Some Tutorial"
git push -u myfork some-tutorial
```

> If you had an error of some kind here, and you think this explanation can be improved, please create an issue, open a Pull Request, or get in touch.

Now go to <https://github.com/JuliaSmoothOptimizers/JSOTutorials.jl> and create a pull request.

## Guidelines

- File and folder names should be `lower-case-separated-by-dashes` (AKA kebab-case);
- Don't forget to add Project.toml, but don't add Manifest.toml.

## Not discussed

The following topics were not included in the short documentation above for lack of time. **You can help by contributing**.

- The other functions inside JSOTutorials.
- Generating other kinds of outputs.
- Converting from Jupyter or Pluto notebooks to the Weave format.
- Online preview.
