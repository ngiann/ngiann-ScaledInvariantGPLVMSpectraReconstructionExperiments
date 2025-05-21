using ForwardNeuralNetworks, Printf, Optim, Plots, JLD2, Random

pythonplot()  

function create_scaling_plot_ducks()

    ctrue  = JLD2.load("scaled_duck_dataset.jld2")["c"]
    cgplvm = JLD2.load("scaleinv_gplvm_scaled_coil_3D.jld2")["c"]
    cppca  = JLD2.load("ppca_scaled_coil_3D.jld2")["c"]
 
    plot(legendfontsize = 22, guidefont = 22, legend = :topleft, tickfontsize = 14)
    scatter!(ctrue, cgplvm, label = "scale invariant GPLVM", markersize=11, color=:cyan)
    scatter!(ctrue, cppca, label = "scale invariant PPCA", markersize=11, color=:red)
    xlabel!("True scaling")
    ylabel!("Recovered scaling")
end