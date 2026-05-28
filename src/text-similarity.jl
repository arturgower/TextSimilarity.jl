abstract type ComparisonMethod end 

struct DirectComparison <: ComparisonMethod 
    shorten_words::Bool
    trim_code::Bool
    remove_comments::Bool
    separate_comments::Bool
    relative_similarity::Bool
end

"""
    DirectComparison(; shorten_words = true, trim_code = true, remove_comments = false, separate_comments = true, relative_similarity = false)

Creates a `DirectComparison` method for comparing strings directly.

# Arguments
- `shorten_words::Bool`: Whether to shorten words in the strings.
- `trim_code::Bool`: Whether to trim code (remove unnecessary characters).
- `remove_comments::Bool`: Whether to remove comments from the strings.
- `separate_comments::Bool`: Whether to separate comments and compare them separately with inverse term frequency.
- `relative_similarity::Bool`: Whether to compute relative similarity.

# Returns
A `DirectComparison` instance.
"""
function DirectComparison(; shorten_words = true, trim_code = true, remove_comments = false, separate_comments = true, relative_similarity = false)
    return DirectComparison(shorten_words, trim_code, remove_comments, separate_comments, relative_similarity)
end

struct DocumentTermsComparison <: ComparisonMethod 
    inverse_term_frequency::Bool
    trim_code::Bool
    remove_comments::Bool
end

"""
    DocumentTermsComparison(; inverse_term_frequency = true, trim_code = true, remove_comments = false)

Creates a `DocumentTermsComparison` method for comparing strings based on document terms.

# Arguments
- `inverse_term_frequency::Bool`: Whether to use inverse term frequency.
- `trim_code::Bool`: Whether to trim code (remove unnecessary characters).
- `remove_comments::Bool`: Whether to remove comments from the strings.

# Returns
A `DocumentTermsComparison` instance.
"""
function DocumentTermsComparison(; inverse_term_frequency = true, trim_code = true, remove_comments = false)
    return DocumentTermsComparison(inverse_term_frequency, trim_code, remove_comments)
end

function shorten_words(strdoc::StringDocument)

    str = deepcopy(strdoc.text)

    # To form a lexicon with just words we will replace the symbols below with spaces
    symbols_vec = ["\n","(", ")", "[", "]", "_" , "=", "*" ,"." , "{", "}", "\$", "^", "&", "!",";"];
    dic_space = Dict(symbols_vec .=> " ")

    for (k, v) in dic_space
        str = replace(str, k => v)
    end

    corpus = Corpus([StringDocument(str)]);

    update_lexicon!(corpus)

    terms = collect(keys(corpus.lexicon));

    ind4 = findall(length.(terms) .> 4);
    term4 = terms[ind4]

    # need to substitute the longer words first
    dic4 = 
    sort(Dict(t => t[nextind(t,0):nextind(t,3)] for t in term4), rev = true)

    ind2to3 = findall(2 .<= length.(keys(corpus.lexicon)) .<= 4);
    term2to3 = terms[ind2to3]

    dic2to3 = sort(Dict(t => t[nextind(t,0):nextind(t,1)] for t in term2to3), rev = true)

    dic = merge(dic4, dic2to3)
    for (k, v) in dic
        strdoc.text = replace(strdoc.text, k => v)
    end    

    return strdoc
end

function trim_and_split(str::String; remove_comments = false)

    # if remove_comments && !trim_code
    #     error("trim_code should be true if you want to remove comments")
    # end

    # str = replace(str, ';' => "" )
    str = replace(str, '_' => "" )
    str = replace(str, ' ' =>"")
    
    if remove_comments
        str = replace(str, r"%[^\n]*" => "")
    end

    s_split = split(str,['\n',';'])

    # remove empty lines
    s_split = s_split[findall(s_split .!= "")]

    # s_split = [string(s,"\n") for s in s_split]

    return [remove_case(s) for s in s_split]
end

function process_strings(strings::Vector{String}, method::DirectComparison)
    strings_vec = if method.trim_code
        trim_and_split.(strings; remove_comments = method.remove_comments)
    else
        [[s] for s in strings]
    end

    stringdocs_vec = [StringDocument.(s_vec) for s_vec in strings_vec]

    if method.shorten_words
        stringdocs_vec = [shorten_words.(docs) for docs in stringdocs_vec]
    end

    strings_vec = map(stringdocs_vec) do strdocs
        str_vec = [replace(doc.text, " " => "") for doc in strdocs]
        str_vec = [replace(s, "\n" => "") for s in str_vec]
        str_vec = [replace(s, "_" => "") for s in str_vec]
    end

    return strings_vec
end

"""
    similarity_matrix(strings::Vector{String}, method::ComparisonMethod)

Builds a pairwise similarity matrix for `strings` using method-specific dispatch.

This function is specialized by method type:
- `similarity_matrix(strings, ::DirectComparison)`: compares processed lines directly.
- `similarity_matrix(strings, ::DocumentTermsComparison)`: compares document-term vectors.

# Arguments
- `strings::Vector{String}`: A vector of strings to compare.
- `method::ComparisonMethod`: The comparison strategy, used for dispatch to a specialized implementation.

# Returns
- `Matrix{Float64}`: Pairwise similarity scores produced by the selected method.
"""
function similarity_matrix(strings::Vector{String}, method::DirectComparison)

    if method.separate_comments
        string_comments = [join((m.match for m in eachmatch(r"%[^\n]*", str)), "\n") for str in strings]
        strings = replace.(strings, r"%[^\n]*" => "")

        sim_comments_matrix = similarity_matrix(
            string_comments,
            DocumentTermsComparison(; inverse_term_frequency = true, trim_code = false, remove_comments = false),
        )
    end

    strings_vec = process_strings(strings, method)

    int_arr = [[Int.(s |> collect) for s in str] for str in strings_vec]

    sim_matrix = [
        begin
            l = min(length(int_arr[i]), length(int_arr[j]))
            if l == 0
                0.0
            else
                sim = map(1:l) do k
                    len = min(length(int_arr[i][k]), length(int_arr[j][k]))
                    if len == 0
                        0.0
                    else
                        dot(int_arr[i][k][1:len],int_arr[j][k][1:len]) / (norm(int_arr[i][k]) * norm(int_arr[j][k]))
                    end    
                end |> sum
                sim / l
            end
        end    
    for i = 1:length(int_arr), j = 1:length(int_arr)]

    # If the comments are completely different, but the code is the same, then the similarity should be 1. If the comments are almost the same, but some different in the code, then the similarity should also be high.
    sim_matrix = if method.separate_comments
        for i in eachindex(sim_matrix)
            if sim_comments_matrix[i] > sim_matrix[i]
                sim_matrix[i] = (sim_comments_matrix[i] + sim_matrix[i]) / 2
            else sim_matrix[i]
            end
        end
        sim_matrix
    else sim_matrix
    end

    sim_matrix = if method.relative_similarity
        similaritytogroup = [
            mean(sim_matrix[i[1],[1:(i[2]-1); (i[2]+1):end]]) / 2  +
            mean(sim_matrix[[1:(i[1]-1); (i[1]+1):end],i[2]]) / 2
        for i in CartesianIndices(sim_matrix)]

        (sim_matrix - similaritytogroup)
    else sim_matrix
    end

    return sim_matrix
end

function similarity_matrix(strings::Vector{String}, method::DocumentTermsComparison)
    
    inverse_term_frequency = method.inverse_term_frequency

    corpus = if method.trim_code
        strings_vec = trim_and_split.(strings; remove_comments = method.remove_comments)
        strings = [string(s_vec...) for s_vec in strings_vec]
        Corpus(StringDocument.(strings))
    else
        Corpus(StringDocument.(strings))
    end    

    remove_case!(corpus)
    prepare!(corpus, strip_punctuation)

    update_lexicon!(corpus)

    m = DocumentTermMatrix(corpus)

    # see the terms identified
    m.terms
    
    tfs =  if inverse_term_frequency
        # Am not sure idf (inverse document frequency) is the best for coding.
        tf_idf(m) |> transpose |> collect
    else    
        # to extract numerical values from this special type we can use 
        dtm(m, :dense) |> transpose |> collect
    end    

    sim_matrix = [
        if norm(tfs[:,i]) == 0 || norm(tfs[:,j]) == 0
            0.0     
        else     
            dot(tfs[:,i],tfs[:,j]) / (norm(tfs[:,i]) * norm(tfs[:,j]))
        end    
    for i = 1:size(tfs,2), j = 1:size(tfs,2)]

    return sim_matrix
end

"""
    text_similarity(strings::Vector{String}, method::ComparisonMethod)

Computes ranked pairwise similarities for `strings` using any `ComparisonMethod`.

`text_similarity` is generic and relies on multiple dispatch: it calls
`similarity_matrix(strings, method)` and therefore automatically uses the
specialized implementation for the concrete method type.

# Arguments
- `strings::Vector{String}`: A vector of strings to compare.
- `method::ComparisonMethod`: The comparison method instance.

# Returns
- `indices::Vector{Vector{Int}}`: Pairs of indices representing similar strings.
- `similarity_vector::Vector{Float64}`: Similarity scores for the pairs.
"""
function text_similarity(strings::Vector{String}, method::ComparisonMethod)

    sim_matrix = similarity_matrix(strings, method)

    indices = [ [i,j] for i = 1:size(sim_matrix,1) for j = (i+1):size(sim_matrix,2)]
    similarity_vector = [sim_matrix[ind...] for ind in indices][:]
    
    # indices_delete = findall(similarity_vector .== -1.0)
    # deleteat!(similarity_vector,indices_delete)
    # deleteat!(indices,indices_delete)

    sort_inds = sortperm(similarity_vector; rev = true)
    indices = indices[sort_inds]
    similarity_vector = similarity_vector[sort_inds]   

    return indices, similarity_vector
end

"""
    group_similar(strings::Vector{String}, method::ComparisonMethod; kws...)

Groups similar strings based on the specified comparison method.

# Arguments
- `strings::Vector{String}`: A vector of strings to group.
- `method::ComparisonMethod`: The comparison method to use.
- `kws...`: Additional keyword arguments.

# Returns
- `group_inds::Vector{Vector{Int}}`: Groups of indices representing similar strings.
- `group_similarities::Vector{Vector{Float64}}`: Similarity scores for the groups.
"""
function group_similar(strings::Vector{String}, method::ComparisonMethod; kws...)
    
    indices, similarity_vector = text_similarity(strings, method);

    return group_similar(indices, similarity_vector; kws...)
end

function group_similar(indices::Vector{Vector{Int}}, similarity_vector::Vector{Float64}; 
        similarity_tolerance::Float64 = 0.985 # ranges from 0 to 1 (identical)
    )

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