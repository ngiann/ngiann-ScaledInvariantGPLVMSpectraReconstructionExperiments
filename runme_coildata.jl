# load coil dataset
using ScaledInvariantGPLVMSpectraReconstruction, COIL20, Random, LinearAlgebra, JLD2, Statistics

rng = MersenneTwister(1);
C   = Diagonal(rand(rng, 62)*2.5 .+ 0.5);

ducks = loadcoil20(;every=8)[1]; 

tr_indices = sort(randperm(rng, 72)[1:62])
te_indices = sort(setdiff(1:72, tr_indices))

Y     = ducks[:,tr_indices] # Y is 256 × 62, i.e. 62 images of 16×16 pixels
Ytest = ducks[:,te_indices] 

σ = var(Y);

S = (σ/10)*randn(size(Y));

Yobs = Y*C + S*C

# filter matrix
B = randn(rng, 30, 256)

##########
# warmup #
##########

let 
    scaleinvariantgplvm(Yobs,S,iterations=3, Q=2, backend = ScaledInvariantGPLVMSpectraReconstruction.LinearBackend());
    scaleinvariantgplvm(Yobs,S,iterations=3, Q=2, backend = ScaledInvariantGPLVMSpectraReconstruction.RecBackend());
    gplvm(Yobs, S, iterations = 3, Q = 2);
end


################
# train models #
################

# run GPLVM and save results
let
    X,rec,res,net,fmin = gplvm(Yobs, S, iterations = 150_000, Q = 2, H = 30, seed = 1);
    JLD2.save("gplvm_coil_2D.jld2", "X", X, "rec", rec, "net", net, "fmin", fmin)
end

# run scale-invariance GPLVM and save results
let
    X,rec,res,net,fmin,c = scaleinvariantgplvm(Yobs,S,iterations = 150_000, Q=2, H = 30, backend = ScaledInvariantGPLVMSpectraReconstruction.LinearBackend(), seed=1);
    JLD2.save("scaleinv_gplvm_coil_2D.jld2", "X", X, "rec", rec, "net", net, "fmin", fmin, "c", c)
end

# run scale-invariance GPLVM and save results
let
    X,rec,res,net,fmin,c = scaleinvariantgplvm(Yobs,S,iterations = 150_000, Q = 2, H = 30, backend = ScaledInvariantGPLVMSpectraReconstruction.RecBackend(), seed=1);
    JLD2.save("scaleinv_positive_gplvm_coil_2D.jld2", "X", X, "rec", rec, "net", net, "fmin", fmin, "c", c)
end

################
# create plots #
################

# create plot for scale-invariant gpplvm
let
    X = JLD2.load("scaleinv_gplvm_coil.jld2", "X")
    plot3D(X[1,:], X[2,:], X[3,:], "-bo")

    # infer latent coordinates of test data
    # infer, getll, pred = scaleinvariantgplvmpredictive(res=res ,net=net, Q=3, D = 256, N = 72)

end
