# load coil dataset
using ScaledInvariantGPLVMSpectraReconstruction, COIL20, Random, LinearAlgebra, JLD2, Statistics

##########
# warmup #
##########

let 
    scaleinvariantgplvm(Yobs,S,iterations=3, Q=2);
    # mlscaleinvariantgplvm(Yobs,S,iterations=3, Q=2, backend = ScaledInvariantGPLVMSpectraReconstruction.RecBackend());
    gplvm(Yobs, S, iterations = 3, Q = 2);
end


################
# train models #
################

# run GPLVM and save results
let
    X,rec,res,net,fmin, = gplvm(Yobs, S, iterations = 30_000, Q = 3, H = 30, seed = 1);
    JLD2.save("gplvm_scaled_coil_2D.jld2", "X", X, "rec", rec, "res", res, "net", net, "fmin", fmin)
end

# run scale-invariance GPLVM and save results
let
    X,rec,res,net,fmin,c,cvar = scaleinvariantgplvm(Yobs, S, iterations = 30_000, Q=3, H = 30, seed=1);
    JLD2.save("scaleinv_gplvm_scaled_coil_2D.jld2", "X", X, "rec", rec, "res", res, "net", net, "fmin", fmin, "c", c, "cvar", cvar)
end

# # run scale-invariance GPLVM and save results
# let
#     X,rec,res,net,fmin,c = mlscaleinvariantgplvm(Yobs,S,iterations = 200_000, Q = 3, H = 30, backend = ScaledInvariantGPLVMSpectraReconstruction.LinearBackend(), seed=1);
#     JLD2.save("mlscaleinv_positive_gplvm_scaled_coil_2D.jld2", "X", X, "rec", rec, "res", res, "net", net, "fmin", fmin, "c", c)
# end
