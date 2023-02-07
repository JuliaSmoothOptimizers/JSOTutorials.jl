module JSOTutorials

# We copied SciML/SciMLTutorials.jl here

using Weave, Pkg, IJulia, InteractiveUtils, Markdown, YAML

repo_directory = joinpath(@__DIR__, "..")
# cssfile = joinpath(@__DIR__, "..", "templates", "skeleton_css.css")
# latexfile = joinpath(@__DIR__, "..", "templates", "julia_tex.tpl")
default_builds = (:github, )

function weave_file(folder, file, build_list = default_builds)
  target = joinpath(repo_directory, "tutorials", folder, file)
  @info("Weaving $(target)")
  set_chunk_defaults!(:line_width => 200)

  if isfile(joinpath(repo_directory, "tutorials", folder, "Project.toml"))
    @info("Instantiating", folder)
    Pkg.activate(joinpath(repo_directory, "tutorials", folder))
    Pkg.instantiate()
    Pkg.build()

    @info("Printing out `Pkg.status()`")
    Pkg.status()
  end

  args = Dict{Symbol, String}(:folder => folder, :file => file)
  if :script ∈ build_list
    println("Building Script")
    dir = joinpath(repo_directory, "script", basename(folder))
    mkpath(dir)
    tangle(target; out_path = dir)
  end
  # if :html ∈ build_list
  #   println("Building HTML")
  #   dir = joinpath(repo_directory, "html", basename(folder))
  #   mkpath(dir)
  #   weave(target, doctype = "md2html", out_path = dir, args = args, css = cssfile, fig_ext = ".svg")
  # end
  # if :pdf ∈ build_list
  #   println("Building PDF")
  #   dir = joinpath(repo_directory, "pdf", basename(folder))
  #   mkpath(dir)
  #   try
  #     weave(target, doctype = "md2pdf", out_path = dir, template = latexfile, args = args)
  #   catch ex
  #     @warn "PDF generation failed" exception = (ex, catch_backtrace())
  #   end
  # end
  if :github ∈ build_list
    println("Building Github Markdown")
    dir = joinpath(repo_directory, "markdown", basename(folder))
    mkpath(dir)
    weave(target, doctype = "github", out_path = dir, args = args)
  end
  if :notebook ∈ build_list
    println("Building Notebook")
    dir = joinpath(repo_directory, "notebook", basename(folder))
    mkpath(dir)
    Weave.convert_doc(target, joinpath(dir, file[1:(end - 4)] * ".ipynb"))
  end
end

function weave_all(build_list = default_builds)
  for folder in readdir(joinpath(repo_directory, "tutorials"))
    folder == "test.jmd" && continue
    weave_folder(folder, build_list)
  end
end

function weave_folder(folder, build_list = default_builds)
  for file in readdir(joinpath(repo_directory, "tutorials", folder))
    # Skip non-`.jmd` files
    if !endswith(file, ".jmd")
      continue
    end

    try
      weave_file(folder, file, build_list)
    catch e
      @error(e)
    end
  end
end

function tutorial_footer(folder = nothing, file = nothing)
  display(
    md"""
## Appendix

These tutorials are a part of the JSOTutorials.jl repository, found at: <https://github.com/JuliaSmoothOptimizers/JSOTutorials.jl>.

""",
  )
  if folder !== nothing && file !== nothing
    display(Markdown.parse("""
    To locally run this tutorial, do the following commands:
    ```
    using JSOTutorials
    JSOTutorials.weave_file("$folder","$file")
    ```
    """))
  end
  display(md"Computer Information:")
  vinfo = sprint(InteractiveUtils.versioninfo)
  display(Markdown.parse("""
  ```
  $(vinfo)
  ```
  """))

  display(md"""
  Package Information:
  """)

  proj = sprint(io -> Pkg.status(io = io))
  mani = sprint(io -> Pkg.status(io = io, mode = Pkg.PKGMODE_MANIFEST))

  md = """
  ```
  $(chomp(proj))
  ```

  And the full manifest:

  ```
  $(chomp(mani))
  ```
  """
  display(Markdown.parse(md))
end

function open_notebooks()
  Base.eval(Main, Meta.parse("import IJulia"))
  weave_all((:notebook,))
  path = joinpath(repo_directory, "notebook")
  IJulia.notebook(; dir = path)
end

"""
    parse_markdown_into_franklin(infile, outfile)

Parses the `infile` returned by Weave into a file that can be understood in Franklin.

Expected:

- `infile`: `markdown/...`
- `outfile`: `parsed/...`
"""
function parse_markdown_into_franklin(infile, outfile)
  @info "Parsing Markdown file into parsed file"
  yaml = YAML.load_file(infile)
  if any(!haskey(yaml, k) for k in ["title", "author", "tags"])
    error("file header is missing some key. It should have title, author and tags")
  end

  open(outfile, "w") do io
    println(io, """
    @def title = "$(yaml["title"])"
    @def showall = true
    @def tags = $(yaml["tags"])

    \\preamble{$(yaml["author"])}
    """)

    code_fence = startswith("```")
    inside_fenced_code = false
    yaml_count = 0
    for line in readlines(infile)
      # Remove YAML header
      if yaml_count < 2
        if line == "---"
          yaml_count += 1
        end
        continue
      end
      if code_fence(line)
        if line == "```"
          if !inside_fenced_code
            line = "```plaintext"
          end
          inside_fenced_code = !inside_fenced_code
        else
          inside_fenced_code = true
        end
      end
      println(io, line)
    end
  end
  @info "Done parsing Markdown file into parsed file"
end

end
