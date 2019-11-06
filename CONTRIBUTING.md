# How to contribute with a tutorial

## TL;DR

Here's the Too Long; Didn't Read version:

- Fork [JSOTutorials.jl](https://github.com/JuliaSmoothOptimizers/JSOTutorials.jl) and clone;
- Choose a **topic** and **title**;
- Create **topic/title/title.jmd** (a Weave.jl file);
- Use `JSOTutorials.conversions` to convert the `.jmd`;
- Create a [GitHub personal token](https://github.com/settings/tokens);
- Enable Travis and add the personal token as variable `GITHUB_AUTH`;
- Create branch `new/short-abrv-title`;
- Commit, push and create a Pull Request;
- Check `https://USER.github.io/JSOTutorials.jl/new/short-abrv-title/`.

## Long version

Greetings, thank you for your interest in contributing to a  **JuliaSmoothOptimizers** tutorial.
This guide will help you to develop your first tutorial and submit it for evaluation.
The first step in the process is deciding what your tutorial is about and whether it fits in the current distribution of topics.
Look into the [root folder of the repository](https://github.com/JuliaSmoothOptimizers/JSOTutorials.jl).
Apart from the `src` and `test` folders, all other folder are topics for created tutorials.
If you believe your tutorial would fit better in a new folder, don't worry, you can create a new one.
Let's denote this the **topic** folder.

Now, look inside the **topic**. You will see a list of folders, one for each tutorial.
Check that the tutorial you want to create hasn't been created yet.
It may also be the case to check whether there is a *Pull Request* open.
Otherwise, decide how you're going to call your tutorial -- let's denote it the **title** of your tutorial.
For organization, both **topic** and **title** need to be *lowercase* and *hyphen-separated*.
Now that you decided the **topic** and **title** of your tutorial, it's time to create it.

First, *fork* the JSOTutorials.jl repository to your account.
This should create a repository of the form `https://github.com/USER/JSOTutorials.jl`.
Notice the **USER**, -- it should be you! -- we will use it later.
In your computer, clone your repository, and enter the folders **topic** and **title**, creating them if necessary.
Inside this folder you will start a [Weave.jl](https://github.com/JunoLab/Weave.jl) file: a file with extension `.jmd`.
If you're not familar with Weave, it may be easier to create a Jupyter Notebook instead, and later convert it to `.jmd`, or to use Atom with specific packages to render the `.jmd` file.
Please refer to the [Weave.jl](https://github.com/JunoLab/Weave.jl) tutorial for more information.
Either way, I will assume you now created `.jmd` files.

It's imperative that you create a file called **title.jmd**, i.e., the file should have the same name as the folder. The header of the file is something like
```yaml
---
title: Some Title
author: Your Name
---

![](../../jso-banner.png)
```
Although `Some Title` doesn't have to be exactly  the same as **title**, it's better to keep it close.
Notice that the logo `jso-banner.png` is included as well.
It's also important that the tutorial is reproducible, so you need to create a environment and keep the `Project.toml` and `Manifest.toml` files.
I recommend something like
```julia
using Pkg
pkg"activate ."
if false # Install these packages:
  pkg"add THEPACKAGESYOUNEED"
end
pkg"instantiate"
```
This part might be improved in the future, but for now it shows how to install the packages, activates the environment and instantiates it.

Work on it until you are satisfied with your tutorial.
Now you should have a new file **title.jmd** inside folder **title** inside of **topic**.
To submit your tutorial, you will need a branch and you will need to create the various formats that we support for our tutorials.
Then, inside **topic/title**, enter
```julia
pkg> activate ../..
julia> using JSOTutorials
julia> JSOTutorials.conversions("title.jmd")
pkg> test
```
This should create `.html`, `.ipynb` and `.jl` files using `Weave.jl`, and then test that they exits.

Now comes the online part. First, go to [Travis-CI](https://travis-ci.org/) and enable travis for `JSOTutorials.jl`.
Then, go to your github and create a [personal token](https://github.com/settings/tokens), enabling the `repo` flag so that travis has access to your repository. 
Back on Travis, go to the settings of `JSOTutorials.jl` and create a new environment variable named `GITHUB_AUTH` and whose value is your personal token. **This should not be public**.
Create a branch (I recommend a name like `new/short-abrv-title`), add the files, commit, push, and create a Pull Request:
```bash
git checkout -b new/short-abrv-title
git add .
git commit -m "Creates Full Title of Tutorial"
git push -u origin new/short-abrv-title
```
This assumes `origin` is your remote. When pushing, `git` shows a link that leads to a pull request. You can use that link directly, or go to GitHub and manually create the Pull Request.

If all went well, you should have one travis running in your repository. Check that it passed, and if not, fix the problem.
When travis passes the test, you should have a webpage with a preview of your tutorial.
The link should be something like
```
https://USER.github.io/JSOTutorials.jl/new/short-abrv-title/
```
**USER** is your GitHub user, and the `new/short-abrv-title` is the branch.
If it doesn't work, check in the package setting that the "GitHub Pages" are being published.

Feel free to open an issue or look for us in the Julia slack channel #smooth-optimizers if this doesn't work.

![](jso-banner.png)
