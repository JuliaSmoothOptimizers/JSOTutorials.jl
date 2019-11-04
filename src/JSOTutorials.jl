module JSOTutorials

using Weave

function conversions(filename)
  @assert filename[end-3:end] == ".jmd"
  Weave.weave(filename,  out_path=:doc, doctype="md2html")
  Weave.tangle(filename, out_path=:doc, informat="markdown")
  Weave.convert_doc(filename, filename[1:end-4] * ".ipynb", format="notebook")
end

end
