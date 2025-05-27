# load coil dataset
using ScaledInvariantGPLVMSpectraReconstruction, Random, LinearAlgebra, JLD2, Statistics
using ScaledInvariantGPLVMSpectraReconstructionSpectralData
using Printf

##########
# warmup #
##########

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

# WARMUP
let 
    local f_tr, σ_tr, f_te, σ_te = split_training_testing_spectra_data()
    scaleinvariantgplvm(f_tr[:,1:10], σ_tr[:,1:10], iterations=3, Q=2)
    gplvm(f_tr[:,1:10], σ_tr[:,1:10], iterations = 3, Q = 2)
    ppca(f_tr[:,1:10], σ_tr[:,1:10], iterations = 3, Q = 2) 
end


################
# train models #
################

# train GPLVM and save results
function train_GPLVM_on_spectra(;seed::Int64 = seed)
    local filename = @sprintf("gplvm_spectra_seed=%d.jld2", seed)
    @printf("Will save result in file |%s|\n\n", filename)
    local f_tr, σ_tr, = split_training_testing_spectra_data()
    local X,rec,res,net,fmin, = gplvm(f_tr, σ_tr, iterations = 30_000, Q = 3, H = 30, seed = seed);
    JLD2.save(filename, "X", X, "rec", rec, "res", res, "net", net, "fmin", fmin)
end

# train scale invariant GPLVM and save results
function train_scale_invariant_GPLVM_on_spectra(;seed::Int64 = seed)
    local filename = @sprintf("scaleinv_gplvm_spectra_seed=%d.jld2", seed)
    @printf("Will save result in file |%s|\n\n", filename)
    local f_tr, σ_tr, = split_training_testing_spectra_data()
    local X,rec,res,net,fmin,c,cvar = scaleinvariantgplvm(f_tr, σ_tr, iterations = 30_000, Q = 3, H = 30, seed = seed);
    JLD2.save(filename, "X", X, "rec", rec, "res", res, "net", net, "fmin", fmin, "c", c, "cvar", cvar)
end


# train scale-invariance PPCA and save results
function train_scale_invariant_ppca_on_spectra(;seed::Int64 = seed)
    local filename = @sprintf("scaleinv_ppca_spectra_seed=%d.jld2", seed)
    @printf("Will save result in file |%s|\n\n", filename)
    local f_tr, σ_tr, = split_training_testing_spectra_data()
    f_tr[isinf.(f_tr)] .= mean(filter(x->~isinf(x), f_tr))
    σ_tr[isinf.(σ_tr)] .= 10000.0 # inflated variance for missing values
    X, res, rec, _, fmin, c = ppca(f_tr, σ_tr; iterations = 30_000, Q=3, seed = 1)
    JLD2.save(filename, "X", X, "rec", rec, "res", res, "fmin", fmin, "c", c)
end
