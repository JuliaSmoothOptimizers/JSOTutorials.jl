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
      # Only require the source markdown and Project.toml to be present
      print("  index.jmd   exists…… ")
      @assert isfile(prefix * ".jmd")
      println("✓")
      
      print("  Project.toml exists…… ")
      @assert isfile(joinpath(rootdir, dir, subdir, "Project.toml"))
      println("✓")
    end
  end
end

tests()
