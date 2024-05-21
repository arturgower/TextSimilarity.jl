abstract type ComparisonMethod end 

struct DirectComparison <: ComparisonMethod 
    shorten_words::Bool
end

function DirectComparison(; shorten_words = true)

    return  DirectComparison(shorten_words)
end

struct DocumentTermsComparion <: ComparisonMethod 
    inverse_term_frequency::Bool
end

function DocumentTermsComparion(; inverse_term_frequency = true)

    return  DocumentTermsComparion(inverse_term_frequency)
end

function shorten_words(strdoc::StringDocument)

    str = deepcopy(strdoc.text)

    # To form a lexicon with just words we will replace the symbols below with spaces
    symbols_vec = ["(", ")", "[", "]", "_" , "=", "*" ,"." , "{", "}", "\$", "^", "&", "!",";"];
    dic_space = Dict(symbols_vec .=> " ")

    for (k, v) in dic_space
        str = replace(str, k => v)
    end

    corpus = Corpus([StringDocument(str)]);

    update_lexicon!(corpus)

    terms = collect(keys(corpus.lexicon));

    ind4 = findall(length.(terms) .> 4);
    term4 = terms[ind4]

    dic4 = Dict(t => t[1:4] for t in term4)

    ind2to3 = findall(2 .<= length.(keys(corpus.lexicon)) .<= 4);
    term2to3 = terms[ind2to3]

    dic2to3 = Dict(t => t[1:2] for t in term2to3)

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

function text_similarity(strings::Vector{String}, method::DirectComparison; 
        trim_code = true, remove_comments = false
    )

    stringdocs = if trim_code
        trimmed_string_document.(strings; remove_comments = remove_comments)
    else
        StringDocument.(strings)
    end

    if method.shorten_words
        stringdocs = shorten_words.(stringdocs);
    end    

    string_vecs = map(stringdocs) do strdoc
        str = replace(strdoc.text, " " => "")
        str = replace(str, "\n" => "")
        str = replace(str, "_" => "")

        Int.(str |> collect)
    end;

    similarity_matrix = [
        if i >= j 
            -1.0
        else
            l = min(length(string_vecs[i]), length(string_vecs[j]))  
            dot(string_vecs[i][1:l],string_vecs[j][1:l]) / (norm(string_vecs[i]) * norm(string_vecs[j]))
        end    
    for i = 1:length(string_vecs), j = 1:length(string_vecs)]

    similarity_vector = similarity_matrix[:]
    indices = [ [i,j] for i = 1:length(string_vecs), j = 1:length(string_vecs)][:]

    indices_delete = findall(similarity_vector .== -1.0)
    deleteat!(similarity_vector,indices_delete)
    deleteat!(indices,indices_delete)

    sort_inds = sortperm(similarity_vector; rev = true)
    indices = indices[sort_inds]
    similarity_vector = similarity_vector[sort_inds]   

    return indices, similarity_vector
end


"""
    text_similarity(strings::Vector{String}, DocumentTermsComparion; trim_code = true, remove_comments = false)

Collects all the terms (or words) in all the strings which I think is then called the lexicon. Then creates a vector for each string with the number of occurences of each term in the lexicon. These vectors are then the rows of the DocumentTermMatrix. We then just compute the distance between the rows.

The options: 
    trim_code = true # removes capitals, semi-colon, and stems words
"""
function text_similarity(strings::Vector{String}, method::DocumentTermsComparion; 
        trim_code = true, remove_comments = false,
        inverse_term_frequency = true
    )

    inverse_term_frequency = method.inverse_term_frequency

    corpus = if trim_code
        Corpus(trimmed_string_document.(strings; remove_comments = remove_comments))
    else
        Corpus(StringDocument.(strings))
    end    

    update_lexicon!(corpus)

    m = DocumentTermMatrix(corpus)

    # see the terms identified
    m.terms
    
    tfs =  if inverse_term_frequency
        # Am not sure idf, which stands for "inverse document frequency" is the best for coding.
        tf_idf(m) |> transpose |> collect
    else    
        # to extract numerical values from this special type we can use 
        dtm(m, :dense) |> transpose |> collect
    end    

    similarity_matrix = [
        if i >= j 
            -1.0
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

    # println("The two most similar documents are")
    # println("Doc 1: \n $(corpus[indices[1][1]].text)")
    # println("Doc 2: \n $(corpus[indices[1][2]].text)")

    return indices, similarity_vector
end

function group_similar(strings::Vector{String}, method::ComparisonMethod; 
        similarity_tolerance::Float64 = 0.985 # ranges from 0 to 1 (identical)
        , kws...
    )

    indices, similarity_vector = text_similarity(strings, method; kws...);

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