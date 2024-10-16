### Distribución Normal(0,λ)
### Autor: Dr. Arturo Erdely
### Versión: 2024-10-15

using Distributions

#=
Una máquina produce cierto componente que debe tener una longitud especificada.
Sea la variable aleatoria X igual al margen de error en dicha longitud y supongamos
que se distribuye Normal(0,λ) donde el parámetro de precisión λ>0 es desconocido 
(y es igual al inverso multiplicativo de la varianza de X). Suponga que no cuenta 
con información a priori y que obtiene una muestra:
=#
x = [0.033, 0.002, -0.019, 0.013, -0.008, -0.0211, 0.009, 0.021, -0.015]
#=
Construya un intervalo de probabilidad:
=#
γ = 0.95
#=
de longitud mínima para el margen de error de dicha componente
=#

#=
La densidad predictiva a posteriori no informativa es t-Student(0, n/Σxi², n)
=#
n = length(x)
ss = sum(x .^ 2)

#=
La función de cuantiles es:
=#
Q(u) = √(ss/n) * quantile(TDist(n), u)

#=
Por simetría de la función de densidad los extremos del intervalo son:
=#
a = Q((1-γ)/2)
b = Q((1+γ)/2)
