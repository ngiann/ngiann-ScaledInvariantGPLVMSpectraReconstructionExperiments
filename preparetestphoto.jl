function preparetestphoto()
     
    redshift, targetspectrum, _, ϕ, σ, u = loadphotometry()

    new_indices = randperm(MersenneTwister(13), 1256)

    # shuffle same way as in function |split_training_testing_spectra_data()|
    redshift = redshift[new_indices]
    targetspectrum = targetspectrum[:,new_indices]
    ϕ = ϕ[:,new_indices]
    σ = σ[:,new_indices]
    
    # return only test data
    return redshift[1001:1256], targetspectrum[:,1001:1256], ϕ[:,1001:1256], σ[:,1001:1256], u

end 