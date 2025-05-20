function ducks_test_ground_truth()

    Ytest = JLD2.load("scaled_duck_dataset.jld2")["Ytest"]

    gt = reduce(hcat, [rot180(reshape(y, 16, 16)) for y in eachcol(Ytest)])

    heatmap(gt, colorbar=false,  yticks=false, xticks=false, color=:greys, size=(2200.0, 280))
    
end