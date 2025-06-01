using JLD2, Statistics, Printf, ScaledInvariantGPLVMSpectraReconstructionSpectralData
using ScaledInvariantGPLVMSpectraReconstruction, Random, LinearAlgebra
using SloanUGRIZFilters

include("preparetestphoto.jl")
include("split_training_testing_spectra_data.jl")


function evaluate_ppca(filename;repeat = 10)

    
    λ_ = loadspectra()[1] # get only wavelength in restframe

    # we work with every second wavelength
    λ = λ_[1:2:end]; @assert(length(λ) == 500)

    z_te, targetspectrum, ϕ_te, σ_te, u = preparetestphoto()

    _, _, spectrum_te, _ = split_training_testing_spectra_data();

    # store here mean squared error
    nmse = zeros(256)

    # load trained scale invariant PPCA
    res = JLD2.load(filename, "res")
    Q   = size(JLD2.load(filename, "X"), 1)
    
    
    # get predictive function for scaled_invariant ppca, call for 0 iterations
    # call like this e.g. infer(B, ϕ, σ; retries = 10)
    # returns reconstructed object and latent coordinate: copt*(W*zopt + b), zopt    
    reconstruct = let
        
        local f_tr, σ_tr  = split_training_testing_spectra_data();

        f_tr[isinf.(f_tr)] .= mean(filter(x->~isinf(x), f_tr))
    
        σ_tr[isinf.(σ_tr)] .= 10000.0 # inflated variance for missing values
    
        ppca(f_tr, σ_tr, res; Q = Q, iterations = 0)[4]

    end
  

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

        # do inference using trained model
        rec, = reconstruct(B, ϕ_te[:,index], σ_te[:,index], retries = repeat)

        # plotting below is useful for verifications purposes. CHECKS OUT ✓
        figure(0);cla();
        plot(u,targetspectrum[:,index],"r",lw=4)   # plot observed spectrum over common grid of observed wavelengths
        plot(λobs, spectrum_te[:,index],"g",lw=2)  # plot test restframe spectrum at observed wavelengths
        plot(λobs, rec,"b",alpha=1,lw=3)           # plot reconstructed restframe spectrum at observed wavelengths
        pause(0.1)

        # find indicices in test spectrum that are not Infs
        indices_to_compare = findall(x -> ~isinf(x), spectrum_te[:,index])

        nmse[index] = normalised_nmse(rec[indices_to_compare],  spectrum_te[indices_to_compare, index])

        display(nmse[1:index])
    end

    nmse

end