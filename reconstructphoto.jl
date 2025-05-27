index = 301
λobs = λ*(1+redshift[index])

a = argmin(abs2.(λobs .- u[1]))
b = argmin(abs2.(λobs .- u[end]))
display((a,b))

Bred = createFilterMatrix(λobs[a:b])[:,2:4]
B = Matrix([zeros(a-1,3); Bred; zeros(500-b,3)]')
display(size(B))

X0,c = infer(B,ϕ[:,index],ones(3),repeat=2);

plot(u,targetspectrum[:,index],"r",lw=2)
plot(λobs[a:b], c*pred(X0)[a:b],"b",alpha=0.8)