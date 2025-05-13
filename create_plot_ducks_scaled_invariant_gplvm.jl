using Plots, JLD2, Random

include("getFakeFilterMatrixB.jl")

function create_plot_ducks_scaled_invariant_gplvm()

    X   = JLD2.load("scaleinv_gplvm_scaled_coil_2D.jld2")["X"]
    res = JLD2.load("scaleinv_gplvm_scaled_coil_2D.jld2")["res"]
    net = JLD2.load("scaleinv_gplvm_scaled_coil_2D.jld2")["net"]

    B = getFakeFilterMatrixB()

    # create filtered images
    σtest = 1e-2

    Φ = B*Ytest + randn(MersenneTwister(1), 256, 10)*σtest

    # get predictive function for scaled_invariant gplvm
    # infer2,getll2,pred2 = scaleinvariantgplvmpredictive(res=res2,net=net2, Q=3, D = 256, N = 72);

    scatter(X[1,:], X[2,:], aspect_ratio=:equal, legend = false)

end