using Statistics
# ŷ is the prediction
# y is the test data item
function normalised_nmse(ŷ, y)

    aux1 = mean(abs2.(ŷ - y))

    aux2 = mean(abs2.(y .- mean(y)))

    aux1 / aux2

end