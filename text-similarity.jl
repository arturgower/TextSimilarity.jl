# Great discussion on the topic:
# https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents

# A website with some details:
# https://makeshiftinsights.com/blog/tf-idf-document-similarity/#:~:text=The%20Method&text=TF%2DIDF%20is%20essentially%20a,inverse%20document%20frequency%20(IDF).

# A much more advanced method would be to use MOSS:
# https://github.com/soachishti/moss.py 


using TextAnalysis, LinearAlgebra

using CSV, DataFrames
using Statistics, LinearAlgebra




# corpus = Corpus([
#     StringDocument("To be or not to be"),
#     StringDocument("To become or not to become"), 
#     StringDocument("there are too many fishies in this pond 2 =3 * 4"), 
#     StringDocument("what if I use the same formula 2 =3 * 4"),
#     StringDocument(" a2 = x * data \n b = a2 + c \n % I wrote this to do something"),
#     StringDocument(" a3 = x * data \n b2 = a3 + c \n % this does that thing")
# ])

function text_similarity(strings::Vector{String}; trim_code = true, remove_comments = false)

    if remove_comments && !trim_code
        error("trim_code should be true if you want to remove comments")
    end    

    if trim_code 
        strings = map(strings) do s
            s = replace(s, ';' => "" )

            s_split = split(s,"\n");
            s_inds = if remove_comments
                findall(s -> !isempty(s) && s[1] != '%', s_split)
            else
                findall(s -> !isempty(s), s_split)
            end    
            string(s_split[s_inds]...)
        end
    end    

    corpus = Corpus([StringDocument(solution) for solution in strings]);

    update_lexicon!(corpus)

    m = DocumentTermMatrix(corpus)

    # see the terms identified
    m.terms

    # to extract numerical values from this special type we can use 
    dtm(m, :dense)

    tfs = tf_idf(m)

    similarity_matrix = [
        if i >= j 
            0.0
        else     
            dot(tfs[i,:],tfs[j,:]) / (norm(tfs[i,:]) * norm(tfs[j,:]))
        end    
    for i = 1:size(tfs,1), j = 1:size(tfs,1)]


    similarity_vector = similarity_matrix[:]
    indices = [ [i,j] for i = 1:size(tfs,1), j = 1:size(tfs,1)][:]

    indices_delete = findall(similarity_vector .≈ 0.0)
    deleteat!(similarity_vector,indices_delete)
    deleteat!(indices,indices_delete)

    sort_inds = sortperm(similarity_vector; rev = true)
    indices = indices[sort_inds]
    similarity_vector = similarity_vector[sort_inds]   

    println("The two most similar documents are")
    println("Doc 1: \n $(corpus[indices[1][1]].text)")
    println("Doc 2: \n $(corpus[indices[1][2]].text)")

    return indices, similarity_vector
end

function group_similar(strings::Vector{String}; 
        similarity_tolerance::Float64 = 0.985 # ranges from 0 to 1 (identical)
        , kws...
    )

    indices, similarity_vector = text_similarity(strings; kws...);


    is = findall(similarity_vector .> similarity_tolerance);
    similarities = deepcopy(similarity_vector[is]);
    indices = indices[is];

    pairs = deepcopy(indices);

    group_inds = Vector{Int}[]

    while !isempty(pairs)

        ind1s = pairs[1]
        group1 = pairs[1]

        while true
            ps = vcat(
                [findall(pair -> any(ind .== pair), pairs) for ind in ind1s]...
            );
            ps = sort(union(ps))

            if !isempty(ps)

                # the possibly new inds
                new_elements = union(vcat(pairs[ps]...))

                # find new elements that have not been searched already
                ind1s = setdiff(new_elements, group1)
    
                # add all new_elements to the whole group
                append!(group1, ind1s)
                group1 = union(group1)

                deleteat!(pairs,ps)

                if ind1s |> isempty break end    

            else break    
            end
        end

        push!(group_inds, group1)
    end

    group_similarities = map(group_inds) do group1
        map(group1) do ind1
        
            ps = findall(ind -> any(ind .== ind1), indices)
            maximum(similarities[ps])
        end    
    end

    return group_inds, group_similarities
end