const site_header = """
<!DOCTYPE html>
<html lang = "en">
<HEAD>
  <meta charset="UTF-8"/>
  <title>JuliaSmoothOptimizers Tutorials</title>
</HEAD>
<BODY>
  <h1>JuliaSmoothOptimizers Tutorials</h1>
"""

const site_footer = """
</BODY>
"""

function generate_site()
  pr = get(ENV, "TRAVIS_PULL_REQUEST", nothing)
  if pr === nothing # Local build
    # Removing "site" since this is a test
    if isdir(joinpath(@__DIR__, "site"))
      rm(joinpath(@__DIR__, "site"), recursive = true)
    end
  elseif pr != "false"
    # This is a PR. Don't generate the site, as it can't be used
    return
  end
  build_dir = joinpath(@__DIR__, "site")
  mkpath(build_dir)

  function titlename(name)
    titlecase(join(split(name, "-"), " "))
  end

  cp(joinpath(@__DIR__, "jso.png"), joinpath(build_dir, "jso.png"))
  cp(joinpath(@__DIR__, "jso-banner.png"), joinpath(build_dir, "jso-banner.png"))
  open(joinpath(build_dir, "index.html"), "w") do io
    println(io, site_header)
    println(io, "<ul>")
    dirs = filter(
      x -> isdir(joinpath(@__DIR__, x)) && x[1] != '.' && !(x in ["site", "src", "test"]),
      readdir(@__DIR__),
    )
    for dir in dirs
      print(io, "<li>")
      print(io, titlename(dir))
      println(io, "</li>")

      println(io, "<ul>")
      subdirs = readdir(joinpath(@__DIR__, dir))
      for subdir in subdirs
        fullpath = joinpath(dir, subdir, subdir)
        print(io, "<li><a href=\"$fullpath.html\">$(titlename(subdir))</a> - Download")
        for ft in ("ipynb", "jl")
          print(io, " <a href=\"$fullpath.$ft\">$ft</a>")
        end
        println(io, "</li>")

        src = joinpath(@__DIR__, dir, subdir)
        dst = joinpath(build_dir, dir, subdir)
        mkpath(dst)
        for file in readdir(src)
          if !(split(file, ".")[end] in ["jmd", "toml"])
            cp(joinpath(src, file), joinpath(dst, file))
          end
        end
      end
      println(io, "</ul>")
    end
    println(io, "</ul>")
    println(io, site_footer)
  end
end

function push_to_gh_pages()
  pr = get(ENV, "TRAVIS_PULL_REQUEST", nothing)
  if pr === nothing
    println("Local build. Not pushing")
    return
  elseif pr != "false"
    println("PR. Not pushing")
    return
  end
  key = get(ENV, "GITHUB_AUTH", nothing)
  if key === nothing
    error("GITHUB_AUTH not found.")
  end

  repo = ENV["TRAVIS_REPO_SLUG"]
  user = split(repo, "/")[1]
  upstream = "https://$user:$key@github.com/$repo"
  run(`git remote add upstream $upstream`)
  run(`git fetch --all`)
  if !success(`git checkout -f -b gh-pages upstream/gh-pages`)
    run(`git checkout --orphan gh-pages`)
    run(`git reset --hard`)
    run(`git commit --allow-empty -m "Initial commit"`)
  end

  dst = ENV["TRAVIS_BRANCH"]
  if dst == "main"
    dst = "."
  else
    # When fixing a PR, the branch folder in gh-pages needs to be removed so it can be created again
    isdir(dst) && rm(dst, recursive = true)
    mkpath(dst)
  end

  for d in readdir("site")
    mv("site/$d", "$dst/$d", force = true)
  end
  rm("site")
  run(`git add $dst`)
  if !success(`git diff --cached --exit-code`) # Don't commit if there are no changes
    run(`git commit -m ":robot: Building tutorials pages"`)
    run(`git push upstream gh-pages`)
  end

  site = "https://$user.github.io/JSOTutorials.jl/" * (dst == "." ? "" : "$dst/") * "index.html"
  println("Here is your site:")
  println("  \033[1;33m$site\033[0m")
end

generate_site()
push_to_gh_pages()
