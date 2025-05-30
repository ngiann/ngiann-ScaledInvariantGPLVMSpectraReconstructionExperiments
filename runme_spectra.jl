# load coil dataset
using ScaledInvariantGPLVMSpectraReconstruction, Random, LinearAlgebra, JLD2, Statistics
using ScaledInvariantGPLVMSpectraReconstructionSpectralData
using Printf

include("split_training_testing_spectra_data.jl")

##########
# warmup #
##########

let 
    local f_tr, σ_tr, = split_training_testing_spectra_data()
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
function train_scale_invariant_ppca_on_spectra(;seed::Int64 = seed, Q = Q)
    local filename = @sprintf("scaleinv_ppca_spectra_Q=%d_seed=%d.jld2", Q, seed)
    @printf("Will save result in file |%s|\n\n", filename)
    local f_tr, σ_tr, = split_training_testing_spectra_data()
    f_tr[isinf.(f_tr)] .= mean(filter(x->~isinf(x), f_tr))
    σ_tr[isinf.(σ_tr)] .= 10000.0 # inflated variance for missing values
    X, res, rec, _, fmin, c = ppca(f_tr, σ_tr; iterations = 30_000, Q=Q, seed = 1)
    JLD2.save(filename, "X", X, "rec", rec, "res", res, "fmin", fmin, "c", c)
end
