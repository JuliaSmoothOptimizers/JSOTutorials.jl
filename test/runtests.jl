function tests()
  rootdir = joinpath(@__DIR__, "..")
  dirs = filter(
    x -> isdir(joinpath(rootdir, x)) && x[1] != '.' && !(x in ["site", "src", "test"]),
    readdir(rootdir),
  )
  for dir in dirs
    subdirs = readdir(joinpath(rootdir, dir))
    for subdir in subdirs
      prefix = joinpath(rootdir, dir, subdir, subdir)
      println("Verifying files in $dir/$subdir")
      for suffix in [".jmd", ".html", ".ipynb", ".jl"]
        spc = " "^(6 - length(suffix))
        print("  $subdir$suffix$spc exists…… ")
        @assert isfile(prefix * suffix)
        println("✓")
      end
    end
  end
end

tests()
