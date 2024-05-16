
"""
    text_similarity(strings::Vector{String}; trim_code = true, remove_comments = false)

Collects all the terms (or words) in all the strings which I think is then called the lexicon. Then creates a vector for each string with the number of occurences of each term in the lexicon. These vectors are then the rows of the DocumentTermMatrix. We then just compute the distance between the rows.
"""
function text_similarity(strings::Vector{String}; 
        trim_code = true, remove_comments = false
    )

    if remove_comments && !trim_code
        error("trim_code should be true if you want to remove comments")
    end    

    if trim_code 
        strings = map(strings) do str
            str = replace(str, ';' => "" )

            # s_split = split(s,"\n");
            if remove_comments
                s_split = split(str,"\n");
                s_inds = findall(s -> !isempty(s) && s[1] != '%', s_split)

                s_split = [string(s,"\n") for s in s_split[s_inds]]

                string(s_split...)[1:end-1]
            else
                str
            end
        end
    end    

    corpus = Corpus([StringDocument(solution) for solution in strings]);

    update_lexicon!(corpus)

    m = DocumentTermMatrix(corpus)

    # see the terms identified
    m.terms

    # to extract numerical values from this special type we can use 
    tfs = dtm(m, :dense) |> transpose |> collect

    # Am not sure idf, which stands for "inverse document frequency" is the best for coding.
    # tfs = tf_idf(m) |> transpose |> collect

    similarity_matrix = [
        if i >= j 
            -1.0
        else     
            # dot(tfs[i,:],tfs[j,:]) / (norm(tfs[i,:]) * norm(tfs[j,:]))
            dot(tfs[:,i],tfs[:,j]) / (norm(tfs[:,i]) * norm(tfs[:,j]))
        end    
    for i = 1:size(tfs,2), j = 1:size(tfs,2)]


    similarity_vector = similarity_matrix[:]
    indices = [ [i,j] for i = 1:size(tfs,2), j = 1:size(tfs,2)][:]

    indices_delete = findall(similarity_vector .== -1.0)
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