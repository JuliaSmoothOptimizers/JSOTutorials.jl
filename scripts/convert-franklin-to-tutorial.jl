# Convert a jl Franklin file to a jmd file
using Literate

repos = [
  "create-a-manual-model",
  "creating-a-jso-compliant-solver",
  "introduction-to-benchmarkprofiles",
  "introduction-to-cutest",
  "introduction-to-linear-operators",
  "introduction-to-quadraticmodels",
  "introduction-to-ripqp",
  "solve-an-optimization-problem-with-ipopt",
  # "solve-pdenlpmodels-with-jsosolvers",
]

for repo in repos
  url_prefix = "https://raw.githubusercontent.com/jso-docs/$repo/main/"
  output_path = joinpath(@__DIR__, "..", "tutorials", repo)
  if !isdir(output_path)
    mkpath(output_path)
  end

  # Copy the Project.toml over, except for the Franklin and NodeJS packages
  project_file = download(url_prefix * "Project.toml")
  open(joinpath(output_path, "Project.toml"), "w") do io
    for line in readlines(project_file)
      if any(contains(x)(line) for x in ["Franklin", "NodeJS"])
        continue
      end
      println(io, line)
    end
  end

  # Convert the .jl Franklin tutorial into a temporary out_file
  index_file = String(download(url_prefix * "index.jl"))
  out_file, _ = mktemp()
  Literate.markdown(
    index_file,
    name=out_file,
    credit=false,
    keep_comments=true,
    execute=false,
    flavor=Literate.FranklinFlavor(),

    codefence="```julia" => "```",
  )

  # Parse the index.md header and glue with the out_file
  open(joinpath(output_path, "index.jmd"), "w") do io
    header_file = download(url_prefix * "index.md")
    println(io, "---")
    for line in readlines(header_file)
      if line |> strip |> length == 0
        continue
      end

      split_line = split(line)
      if startswith("\\preamble")(line)
        author = split(line, r"[{}]")[2]
        println(io, "author: \"$author\"")
      elseif length(split_line) == 1
        continue
      elseif split_line[2] == "showall"
        continue
      elseif split_line[1] == "@def"
        println(io, split_line[2] * ": " * join(split_line[4:end], " "))
      end
    end
    println(io, "---\n")
    println(io, read(out_file * ".md", String))
  end
end
