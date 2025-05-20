# load coil dataset
using ScaledInvariantGPLVMSpectraReconstruction, COIL20, Random, LinearAlgebra, JLD2, Statistics

##########
# warmup #
##########

let 
    Yobs, Sobs = JLD2.load("scaled_duck_dataset.jld2", "Yobs", "Sobs")
    scaleinvariantgplvm(Yobs, Sobs,iterations=3, Q=2)
    gplvm(Yobs, Sobs, iterations = 3, Q = 2)
    ppca(Yobs, Sobs, iterations = 3, Q = 2) 
end


################
# train models #
################

# run GPLVM and save results
let
    Yobs, Sobs = JLD2.load("scaled_duck_dataset.jld2", "Yobs", "Sobs")
    X,rec,res,net,fmin, = gplvm(Yobs, Sobs, iterations = 30_000, Q = 3, H = 30, seed = 1);
    JLD2.save("gplvm_scaled_coil_3D.jld2", "X", X, "rec", rec, "res", res, "net", net, "fmin", fmin)
end

# run scale-invariance GPLVM and save results
let
    Yobs, Sobs = JLD2.load("scaled_duck_dataset.jld2", "Yobs", "Sobs")
    X,rec,res,net,fmin,c,cvar = scaleinvariantgplvm(Yobs, Sobs, iterations = 30_000, Q=3, H = 30, seed=1);
    JLD2.save("scaleinv_gplvm_scaled_coil_3D.jld2", "X", X, "rec", rec, "res", res, "net", net, "fmin", fmin, "c", c, "cvar", cvar)
end

# run scale-invariance PPCA and save results
let
    Yobs, Sobs = JLD2.load("scaled_duck_dataset.jld2", "Yobs", "Sobs")
    X, res, rec, _, fmin, c = ppca(Yobs, Sobs; iterations = 30_000, Q=3, seed = 1)
    JLD2.save("ppca_scaled_coil_3D.jld2", "X", X, "rec", rec, "res", res, "fmin", fmin, "c", c)
end
