using ForwardNeuralNetworks, Printf, Optim, Plots, JLD2, Random

pythonplot()  

include("getFakeFilterMatrixB.jl")

function create_plot_ducks_scaled_invariant_gplvm()

    X   = JLD2.load("scaleinv_gplvm_coil_2D.jld2")["X"]
    res = JLD2.load("scaleinv_gplvm_coil_2D.jld2")["res"]
    net = JLD2.load("scaleinv_gplvm_coil_2D.jld2")["net"]

    B = getFakeFilterMatrixB()

    # create filtered images
    σtest = 1e-2
    Σ = randn(MersenneTwister(1), 30, 10)*σtest

    Φ = B*Ytest + Σ

    # # get predictive function for scaled_invariant gplvm
    # infer = scaleinvariantgplvmpredictive(res=res,net=net, Q=3, D = 256, N = 62)[1]

    # Xtest = map(1:10) do n

    #     infer(B, Φ[:,[n]], Σ[:,[n]]; repeat=10, seed=1)[1]

    # end

    plot(X[1,:], X[2,:], X[3,:], aspect_ratio=:equal, legend = false, marker = :circle)
    for n in 1:2:62
        annotate!(X[1,n], X[2,n], X[3,n], @sprintf("%d",  n))    
    end

    # for n in 1:10
    #     scatter!([Xtest[n][1]], [Xtest[n][2]], [Xtest[n][3]], marker = :star, color=:red, markersize=12)
    # end

end