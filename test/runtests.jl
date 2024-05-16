# using TextSimilarity
# using Test, Random

# include("../src/text-similarity.jl")

@testset "TextSimilarity.jl" begin

    len = 400;
    str = randstring(len) |> collect;

    files_len = 100;

    str_vec = [str for i = 1:files_len];

    # make more and more modifictions
    inds = 1:len
    strings = map(1:files_len) do i
        str_vec[i][rand(inds,4i)] = collect(randstring(4i))

        string(str_vec[i]...)
    end

    indices, similarity_vector = text_similarity(strings);
    
    # The elements which are most similar to all others are the first ones. So the list below should be approximately decreasing
    element_similarities = map(1:100) do i 
        inds = findall(ind -> any(i .== ind), indices);
        similarity_vector[inds] |> maximum
    end
    
    @test norm(sort(element_similarities, rev=true) - element_similarities) / norm(element_similarities) < 0.1


    @test all(element_similarities[1:3] .> 0.5)
    @test all(element_similarities[end-3:end] .< 0.1)
end
