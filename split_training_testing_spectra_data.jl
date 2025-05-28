function split_training_testing_spectra_data()

    _λ, f, σ = loadspectra()

    # replaces NaNs with Infs
    f[isnan.(f)] .= Inf;
    σ[isnan.(σ)] .= Inf;

    # shuffle dataset just to make sure that there is no particular ordering
    new_indices = randperm(MersenneTwister(13), 1256)
    f = f[:,new_indices]
    σ = σ[:,new_indices]

    @printf("Returning:\n");
    @printf("1st argument: training 500×1000 fluxes\n")
    @printf("2nd argument: training 500×1000 std errors\n")
    @printf("3rd argument: testing 500×256 fluxes\n")
    @printf("4th argument: testing 500×256 standard errors.\n")

    f_tr = f[1:2:end, 1:1000]
    σ_tr = σ[1:2:end, 1:1000]
    f_te = f[1:2:end, 1001:1256]
    σ_te = σ[1:2:end, 1001:1256]

    return f_tr, σ_tr, f_te, σ_te

end