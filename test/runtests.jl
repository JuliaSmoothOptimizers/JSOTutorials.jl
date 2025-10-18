function tests()
  rootdir = joinpath(@__DIR__, "..")
  dirs = filter(
    x -> isdir(joinpath(rootdir, x)) && x[1] != '.' && !(x in ["site", "src", "test"]),
    readdir(rootdir),
  )
  for dir in dirs
    subdirs = filter(y -> isdir(joinpath(rootdir, dir, y)), readdir(joinpath(rootdir, dir)))
    for subdir in subdirs
      prefix = joinpath(rootdir, dir, subdir, "index")
      println("Verifying files in $dir/$subdir")
      # Require the source markdown to be present
      print("  index.jmd   exists…… ")
      @assert isfile(prefix * ".jmd")
      println("✓")

      # Optional: ensure each tutorial folder has a Project.toml for deps
      print("  Project.toml exists…… ")
      @assert isfile(joinpath(rootdir, dir, subdir, "Project.toml"))
      println("✓")
    end
  end
end

tests()
