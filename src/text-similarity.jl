abstract type ComparisonMethod end 

struct DirectComparison <: ComparisonMethod 
    shorten_words::Bool
    trim_code::Bool
    remove_comments::Bool
    relative_similarity::Bool
end

"""
    DirectComparison(; shorten_words = true, trim_code = true, remove_comments = false, relative_similarity = false)

Creates a `DirectComparison` method for comparing strings directly.

# Arguments
- `shorten_words::Bool`: Whether to shorten words in the strings.
- `trim_code::Bool`: Whether to trim code (remove unnecessary characters).
- `remove_comments::Bool`: Whether to remove comments from the strings.
- `relative_similarity::Bool`: Whether to compute relative similarity.

# Returns
A `DirectComparison` instance.
"""
function DirectComparison(; shorten_words = true, trim_code = true, remove_comments = false, relative_similarity = false)
    return DirectComparison(shorten_words, trim_code, remove_comments, relative_similarity)
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

function trimmed_string_document(str::String; remove_comments = false)

    # if remove_comments && !trim_code
    #     error("trim_code should be true if you want to remove comments")
    # end

    str = replace(str, ';' => "" )
    str = replace(str, '_' => "" )
    
    s_split = split(str,"\n");

    # remove empty lines
    s_split = s_split[findall(s_split .!= "")]

    if remove_comments 
        s_inds =  findall(s -> !isempty(s) && s[1] != '%', s_split)
        s_split = s_split[s_inds]
    end    

    s_split = [string(s,"\n") for s in s_split]

    str_doc = StringDocument(string(s_split...)[1:end-1])

    remove_case!(str_doc)
    # stem!(str_doc)

    return str_doc
end

function process_strings(strings::Vector{String}, method::DirectComparison)
    stringdocs = if method.trim_code
        trimmed_string_document.(strings; remove_comments = method.remove_comments)
    else
        StringDocument.(strings)
    end

    if method.shorten_words
        stringdocs = shorten_words.(stringdocs);
    end    

    strings = map(stringdocs) do strdoc
        str = replace(strdoc.text, " " => "")
        str = replace(str, "\n" => "")
        str = replace(str, "_" => "")
    end

    return strings
end

function text_similarity(strings::Vector{String}, method::DirectComparison)

    strings = process_strings(strings, method)

    string_vecs = [Int.(str |> collect) for str in strings]

    similarity_matrix = [
        begin
            l = min(length(string_vecs[i]), length(string_vecs[j]))
            if l == 0
                0.0
            else    
                dot(string_vecs[i][1:l],string_vecs[j][1:l]) / (norm(string_vecs[i]) * norm(string_vecs[j]))
            end    
        end    
    for i = 1:length(string_vecs), j = 1:length(string_vecs)]

    similarity_vector = if method.relative_similarity
        similaritytogroup = [
            mean(similarity_matrix[i[1],[1:(i[2]-1); (i[2]+1):end]]) / 2  +
            mean(similarity_matrix[[1:(i[1]-1); (i[1]+1):end],i[2]]) / 2
        for i in CartesianIndices(similarity_matrix)]

        (similarity_matrix - similaritytogroup)[:]
    else similarity_matrix[:]    
    end

    indices = [ [i,j] for i = 1:length(string_vecs), j = 1:length(string_vecs)][:]

    # indices_delete = findall(similarity_vector .== -1.0)
    indices_delete = findall([ij[1] >= ij[2] for ij in indices])
    deleteat!(similarity_vector,indices_delete)
    deleteat!(indices,indices_delete)

    sort_inds = sortperm(similarity_vector; rev = true)
    indices = indices[sort_inds]
    similarity_vector = similarity_vector[sort_inds]

    return indices, similarity_vector
end

"""
    text_similarity(strings::Vector{String}, method::DocumentTermsComparison)

Computes the similarity between strings using the `DocumentTermsComparison` method.

# Arguments
- `strings::Vector{String}`: A vector of strings to compare.
- `method::DocumentTermsComparison`: The comparison method to use.

# Returns
- `indices::Vector{Vector{Int}}`: Pairs of indices representing similar strings.
- `similarity_vector::Vector{Float64}`: Similarity scores for the pairs.
"""
function text_similarity(strings::Vector{String}, method::DocumentTermsComparison)

    inverse_term_frequency = method.inverse_term_frequency

    corpus = if method.trim_code
        Corpus(trimmed_string_document.(strings; remove_comments = method.remove_comments))
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

    similarity_matrix = [
        if i >= j 
            -1.0
        elseif norm(tfs[:,i]) == 0 || norm(tfs[:,j]) == 0
            0.0     
        else     
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