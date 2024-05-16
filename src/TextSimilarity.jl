# Great discussion on the topic:
# https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents

# A website with some details:
# https://makeshiftinsights.com/blog/tf-idf-document-similarity/#:~:text=The%20Method&text=TF%2DIDF%20is%20essentially%20a,inverse%20document%20frequency%20(IDF).

# A website with a Julia example:
# https://juliasilge.com/blog/term-frequency-tf-idf/

# A much more advanced method would be to use MOSS:
# https://github.com/soachishti/moss.py 

module TextSimilarity

export text_similarity, group_similar

using TextAnalysis, LinearAlgebra
using Statistics, Random

include("text-similarity.jl")

end
