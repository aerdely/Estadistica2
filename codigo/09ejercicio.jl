### Distribución de probabilidad Poisson
### Autor: Dr. Arturo Erdely
### Versión: 2024-10-08


## Cargar paquetes y código externo necesarios:
begin
    using Distributions, Plots
    include("06EDA.jl") 
end

#=
Supongamos que la cantidad de autos que llegan a la caseta de cobro
de una autopista los días viernes de 5 a 8 p.m se puede modelizar
mediante la familia paramétrica Poisson. Hacemos dos preguntas al
encargado de la caseta: ¿Como cuántos autos llegan en promedio por
minuto a la caseta? A lo cual nos responde que 5. Tomando en cuenta
que el dato anterior es una apreciación subjetiva ¿Con probabilidad
95% cuál cree usted que sería en el mayor de los casos el número
promedio de autos por minuto? A lo cual nos responde que 12.

Utilizando una distribución conjugada especifique la distribución a
priori del parámetro con base en la información que se tiene. Con ella
calcule el valor esperado del parámetro así como la probabilidad de que
dicho parámetro sea mayor que 8.
=#

## Distribución y cálculos a priori 

begin
    g(β) = cdf(Gamma(900/β, β), 12*180) - 0.95
    β = range(300, 600, length = 1_000);
    plot(β, g.(β), lw = 3, legend = false, xlabel = "β", ylabel = "g(β)")
    hline!([0.0], color = :red)
end

h(β) = g(β[1])^2 # claramente alcanzaría un mínimo para un β ∈ [400,500]
minh = EDA(h, [400], [500])
vline!([minh.x[1]], color = :red) # comprobando en la gráfica

β0 = 1/minh.x[1] # por la parametrización de la densidad Gamma que usa Distributions.jl
α0 = 900*β0
cdf(Gamma(α0,1/β0), 12*180) # comprobando que P(λ ≤ 12×180) = 0.95
mean(Gamma(α0,1/β0)) # comprobando que E(λ) = 5×180 = 900
α0/β0 # comprobando que E(λ) = α0/β0 = 900
# P(λ > 8×180)
1 - cdf(Gamma(α0,1/β0), 8*180)


#=
Supongamos ahora que procedemos a tomar una muestra aleatoria consistente en
contar el número de autos que llegan a la caseta de 5 a 8 p.m durante cinco
días viernes, y obtenemos {679,703,748,739,693}. Obtenga la distribución a
posteriori del parámetro parámetro y calcule su valor esperado, así como la
probabilidad de que el parámetro sea mayor que 8. Compare con el inciso a).
Grafique en una misma imagen la distribución a priori, y la distribución a
posteriori con el primer dato, con los primeros dos y así sucesivamente hasta
abarcar toda la información muestral.
=#

xobs = [679,703,748,739,693] # muestra observada
n = length(xobs) # tamaño de muestra
post = Gamma(α0+sum(xobs), 1/(β0+n)) # distribución a posteriori
mean(post) # E(λ|xobs)
1 - cdf(post, 8*180) # P(λ > 8×180|xobs)

begin
    xx = range(0, 2200, length = 1_000);
    plot(xx, pdf(Gamma(α0,1/β0), xx), lw = 2, label = "a priori")
    xaxis!("λ")
    yaxis!("π(λ)")
end

begin
    yaxis!("p(λ|xobs)")
    for i ∈ 1:n 
        plot!(xx, pdf(Gamma(α0+sum(xobs[1:i]), 1/(β0+i)), xx), lw = 2, label = "n = $i")
    end
    current()
end


#= 
Ahora supongamos que la percepción del responsable de la caseta de cobro cambió
y ahora cree que con probabilidad 95% el número promedio de autos por minuto
no pasa de 8. 

Suponiendo que como información muestral solo se tiene xobs=[679,703], utilizando
una función de pérdida cuadrática obtenga la estimación puntual correspondiente
para el parámetro λ.
=#

## Recalibrando distribución a priori 
begin
    g(β) = cdf(Gamma(900/β, β), 8*180) - 0.95
    β = range(80, 120, length = 1_000);
    plot(β, g.(β), lw = 3, legend = false, xlabel = "β", ylabel = "g(β)")
    hline!([0.0], color = :red)
end

h(β) = g(β[1])^2 # claramente alcanzaría un mínimo para un β ∈ [90,100]
minh = EDA(h, [90], [100])
vline!([minh.x[1]], color = :red) # comprobando en la gráfica
β0 = 1/minh.x[1] # por la parametrización de la densidad Gamma que usa Distributions.jl
α0 = 900*β0
cdf(Gamma(α0,1/β0), 8*180) # comprobando que P(λ ≤ 8×180) = 0.95
mean(Gamma(α0,1/β0)) # comprobando que E(λ) = 5×180 = 900
α0/β0 # comprobando que E(λ) = α0/β0 = 900


## Estimación puntual

post = Gamma(α0+sum(xobs[1:2]), 1/(β0+2)) # distribución a posteriori
mean(post) # E(λ|xobs)


## Estimación por intervalo

γ = 0.95
ℓ(a) = quantile(post, γ + cdf(post, a[1])) - a[1]
amax = quantile(post, 1-γ)
aa = range(0.001, amax, length = 1_000);
plot(aa, ℓ.(aa), legend = false, xlabel = "a", ylabel = "longitud intervalo")

aa = range(600, amax, length = 1_000);
plot(aa, ℓ.(aa), legend = false, xlabel = "a", ylabel = "longitud intervalo")

# claramente la longitud se minimiza para algún a ∈ [650,660]
minlong = EDA(ℓ, [650], [660])
aopt = minlong.x[1]
vline!([aopt], color = :red) # comprobando mínimo en la gráfica

b(a) = quantile(post, γ + cdf(post, a))
bopt = b(aopt) # extremo superior del intervalo
println("[$aopt,$bopt] es la estimación por intervalo de probabilidad $γ para λ.")


## Contraste de hipótesis

# distribución predictiva a posteriori
pred = NegativeBinomial(α0+sum(xobs[1:2]), (β0+2)/(β0+3))

# probabilidad de cada hipótesis
pH1 = cdf(pred, 690)
pH2 = cdf(pred,750) - cdf(pred, 690)
pH3 = 1 - cdf(pred, 750)
pH = [pH1, pH2, pH3]
sum(pH)

# matriz de pérdidas
L = [1500 5000 8500; 3000 3000 6500; 4500 4500 4500]

# vector de pérdidas esperadas
EL = zeros(3);
for i ∈ 1:3 
    EL[i] = sum(L[i, :] .* pH)
end
EL
iopt = findmin(EL)
println("Escoger la hipótesis $(iopt[2])")
