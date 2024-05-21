using TextSimilarity
using Test, Random, LinearAlgebra

@testset "TextSimilarity.jl" begin

    symbols_string = "\n ( ) [ ] _ , = * . = { } \$ ^ & ! \n";

    len = 400;
    str1 = randstring(len - 100) |> collect;
    str2 = randstring(symbols_string, 100) |> collect;
    inds = rand(1:len,len)

    str = randstring(len) |> collect;
    str[inds[1:(len-100)]] = str1;
    str[inds[(len-99):end]] = str2;

    files_len = 100;

    str_vec = [str for i = 1:files_len];

    # make more and more modifictions
    inds = 1:len
    strings = map(1:files_len) do i
        str_vec[i][rand(inds,3i)] = collect(randstring(3i))
        str_vec[i][rand(inds,i)] = collect(randstring(symbols_string,i))

        string(str_vec[i]...)
    end

    indices, similarity_vector = text_similarity(strings; inverse_term_frequency = false);

    # check that the values in similarity_vector do correspond to the pairs in indices
    inds, sims = text_similarity(strings[indices[1]]; inverse_term_frequency = false) 

    @test sims[1] ≈ similarity_vector[1]
    
    # The elements which are most similar to all others are the first ones. So the list below should be approximately decreasing
    element_similarities = map(1:100) do i 
        inds = findall(ind -> any(i .== ind), indices);
        similarity_vector[inds] |> maximum
    end
    
    @test norm(sort(element_similarities, rev=true) - element_similarities) / norm(element_similarities) < 0.1

    @test all(element_similarities[1:3] .> 0.7)


    group_inds, group_similarities = group_similar(strings; similarity_tolerance = 0.92, inverse_term_frequency = false);

    @test sort(group_inds[1][1:3]) == [1,2,3];


    indices, similarity_vector = text_similarity(strings; inverse_term_frequency = true);

    group_inds, group_similarities = group_similar(strings; similarity_tolerance = 0.7, inverse_term_frequency = true);

    @test group_inds[1][1:3] == [1,2,3];
end
