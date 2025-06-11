using JLD2, Statistics, Printf, ScaledInvariantGPLVMSpectraReconstructionSpectralData
using ScaledInvariantGPLVMSpectraReconstruction, Random, LinearAlgebra
using SloanUGRIZFilters
using DelimitedFiles

include("preparetestphoto.jl")
include("split_training_testing_spectra_data.jl")
include("normalised_nmse.jl")

function evaluate_scale_invariant_gplvm(filename;repeat = 10)

    
    λ_ = loadspectra()[1] # get only wavelength in restframe

    # we work with every second wavelength
    λ = λ_[1:2:end]; @assert(length(λ) == 500)

    z_te, targetspectrum, ϕ_te, σ_te, u = preparetestphoto()

    _, _, spectrum_te, _ = split_training_testing_spectra_data();

    # store here mean squared error
    nmse = zeros(256)

    # load trained scale invariant GPLVM
    res, net = JLD2.load(filename, "res","net")
    infer,_,pred = scaleinvariantgplvmpredictive(res=res,net=net, Q=3, D = 500, N = 1000);

    # for each test data item
    for index in 1:256

        B = readdlm(@sprintf("test/B_%d.csv", index))

        @assert(size(B) == (3, 500))

        # do inference using trained model
        X0, c = infer(B, ϕ_te[:,index], σ_te[:,index], repeat = repeat)

        # reconstructed spectrum corresponding to observed interval
        # rec_a_b = c*pred(X0)[a:b]
        # reconstructed spectrum in restframe
        rec = c*pred(X0)

        # plotting below is useful for verifications purposes. CHECKS OUT ✓
        figure(0);cla();
        # shift grid of restframe spectra
        λobs = λ*(1 + z_te[index])
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