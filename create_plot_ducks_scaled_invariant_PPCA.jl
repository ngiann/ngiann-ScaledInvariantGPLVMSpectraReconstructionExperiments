using ForwardNeuralNetworks, Printf, Optim, Plots, JLD2, Random

pythonplot()  

include("getFakeFilterMatrixB.jl")


function create_plot_ducks_scaled_invariant_ppca()

    X   = JLD2.load("ppca_scaled_coil_3D.jld2")["X"]
    res = JLD2.load("ppca_scaled_coil_3D.jld2")["res"]
 
    Yobs = JLD2.load("scaled_duck_dataset.jld2")["Yobs"]
    Sobs = JLD2.load("scaled_duck_dataset.jld2")["Sobs"]

    Ytest = JLD2.load("scaled_duck_dataset.jld2")["Ytest"]
    tr_indices = JLD2.load("scaled_duck_dataset.jld2")["tr_indices"]
    te_indices = JLD2.load("scaled_duck_dataset.jld2")["te_indices"]

    # create filtered images
    σtest = 1e-2
    Stest = randn(MersenneTwister(1), 30, 10)*σtest

    Φ = B*Ytest + Stest

    # get predictive function for scaled_invariant ppca, call for 0 iterations
    # call like this e.g. infer(B, ϕ, σ; retries = 10)
    # returns reconstructed object and latent coordinate: copt*(W*zopt + b), zopt    
    infer = ppca(Yobs, Sobs, res; Q = 3, iterations = 0)[4]
  

    Xtest = map(1:10) do n

        infer(B, Φ[:,n], σtest*ones(30); retries=100)[2]

    end

    XtestMatrix = reduce(hcat, Xtest)

    offset = 0.03

    plot3d(X[1,:], X[2,:], X[3,:], aspect_ratio=:equal, legend = false, linewidth=0.5   )
    scatter3d!(X[1,:], X[2,:], X[3,:], markersize=15, color = :blue)
    for n in 1:62
        # scatter3d!([X[1,n]], [X[2,n]], [X[3,n]], marker = :circle, color=:blue, markersize=12, depthshade = true)
        annotate!(X[1,n]+offset, X[2,n]+offset, X[3,n]+offset, (@sprintf("%d",  tr_indices[n]), :black, 13), alpha=1) 
    end

    scatter!(XtestMatrix[1,:],XtestMatrix[2,:],XtestMatrix[3,:], markersize=15, color = :red)
        
    for n in 1:10
        annotate!(Xtest[n][1]+offset, Xtest[n][2]+offset, Xtest[n][3]+offset, (@sprintf("%d",  te_indices[n]), :black, 15))  
    end
    
end


function ppca_duck_reconstructions()

    res = JLD2.load("ppca_scaled_coil_3D.jld2")["res"]
 
    Yobs = JLD2.load("scaled_duck_dataset.jld2")["Yobs"]
    Sobs = JLD2.load("scaled_duck_dataset.jld2")["Sobs"]

    Ytest = JLD2.load("scaled_duck_dataset.jld2")["Ytest"]

    # create filtered images
    σtest = 1e-2
    Stest = randn(MersenneTwister(1), 30, 10)*σtest

    Φ = B*Ytest + Stest

    # get predictive function for scaled_invariant ppca, call for 0 iterations
    # call like this e.g. infer(B, ϕ, σ; retries = 10)
    # returns reconstructed object and latent coordinate: copt*(W*zopt + b), zopt    
    infer = ppca(Yobs, Sobs, res; Q = 3, iterations = 0)[4]
  
    recs = map(1:10) do n

        rot180(reshape(infer(B, Φ[:,n], σtest*ones(30); retries=100)[1],16,16))

    end

    heatmap(reduce(hcat, recs), colorbar=false, yticks=false, xticks=false, color=:greys, size=(2200.0, 280))
    
end


function mse_scale_invariant_ppca_duck_reconstructions()

    res = JLD2.load("ppca_scaled_coil_3D.jld2")["res"]
    
    Yobs = JLD2.load("scaled_duck_dataset.jld2")["Yobs"]
    Sobs = JLD2.load("scaled_duck_dataset.jld2")["Sobs"]

    Ytest = JLD2.load("scaled_duck_dataset.jld2")["Ytest"]

    B = getFakeFilterMatrixB()

    # create filtered images
    σtest = 1e-2
    Stest = randn(MersenneTwister(1), 30, 10)*σtest

    Φ = B*Ytest + Stest

    # get predictive function for gplvm
    infer = ppca(Yobs, Sobs, res; Q = 3, iterations = 0)[4]

    recs = map(1:10) do n

        infer(B, Φ[:,n], σtest*ones(30); retries=100)[1]

    end
    
    mse = [sum(abs2, recs[i] - Ytest[:,i]) for i in 1:10]
    
    return mse#, recs, Ytest

end