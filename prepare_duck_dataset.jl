using Random, JLD2, LinearAlgebra, COIL20, Statistics

let
        
    rng = MersenneTwister(1);

    #################
    # prepare data #
    #################

    C   = Diagonal(rand(rng, 62)*2.5 .+ 0.5);

    ducks = loadcoil20(;every=8)[1]; 

    tr_indices = sort(randperm(rng, 72)[1:62])
    te_indices = sort(setdiff(1:72, tr_indices))

    Y     = ducks[:,tr_indices] # Y is 256 × 62, i.e. 62 images of 16×16 pixels
    Ytest = ducks[:,te_indices] 

    σ = var(Y);

    Sobs = (σ/10)*randn(size(Y))*C;

    Yobs = Y*C + Sobs


    JLD2.save("scaled_duck_dataset.jld2","Y",Y,"Ytest",Ytest,"Sobs",Sobs,"Yobs",Yobs,"tr_indices",tr_indices,"te_indices",te_indices,"c",diag(C))

end
