using JLD2, Statistics, Printf, ScaledInvariantGPLVMSpectraReconstructionSpectralData
using ScaledInvariantGPLVMSpectraReconstruction, Random, LinearAlgebra
using SloanUGRIZFilters
using DelimitedFiles

include("preparetestphoto.jl")
include("split_training_testing_spectra_data.jl")


function create_test_photometric_matrices_B()

    λ_ = loadspectra()[1] # get only wavelength in restframe

    # we work with every second wavelength
    λ = λ_[1:2:end]; @assert(length(λ) == 500)

    z_te, _, _, _, u = preparetestphoto()

    # for each test data item
    for index in 1:256

        # shift grid of restframe spectra
        λobs = λ*(1 + z_te[index])

        # find interval that falls inside observed interval
        a = argmin(abs2.(λobs .- u[1])) # this is an index, not a wavelength value
        b = argmin(abs2.(λobs .- u[end])) # this is an index, not a wavelength value
        @printf("[%.3f, %.3f]\n", λobs[a], λobs[b])

        # create filter matrix using only g,r,i filters
        Bred = createFilterMatrix(λobs[a:b])[:,2:4]
        B = Matrix([zeros(a-1,3); Bred; zeros(500-b,3)]')
        @assert(size(B) == (3, length(λobs)))

        # for each index, save filter matrix, test spectrum, target photometry and its noise     
        writedlm(@sprintf("test/B_%d.csv", index), B)

    end

end