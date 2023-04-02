School project by SICOT Guillaume #ENSTA Bretagne

Code produced by VAN STRAATEN Clément & DUVAL Olivier

# DS4_buoy_current_field
In this project we use EnKF or Particle filter to retrieve position of a drifting buoy. In fact, the localisation is carried out thanks to a terrain-based method which requires that the bathymetry is known, we also use current speed measurement to improve the localization. So we need some information about the current field. The buoy is supposed to stay at a constant depth.

Our work is the first step of a more general which is to estimate the current advecting a buoy without GNSS data. This first step need a less complex model and state vector. We didn't have the time to perform current estimation.

## Theoretical case and bathymetry

The current is considered to be a stationnary field with a stream function $$\psi(x, y) = L (\cos x +\cos y)$$
And the following equation represents a field with alternating of cyclonic and anticyclonic gyres:

$$\begin{pmatrix} 
u \\
v
\end{pmatrix}= \begin{pmatrix}
\frac{\partial \psi}{\partial y} \\
-\frac{\partial \psi}{\partial x}
\end{pmatrix}$$


The bathymetry is given by an analytical function. The function define for the bathymetry is given by 
$$g(x,y) = -30+15*(1-e^{-\frac{x^2 + y^2}{100}})+\sin(x + y)$$

## Dynamical model
The advection of the buoy by th current is modeled by 2D advection model of a particle (Crisanti *et al.* 92).

$$\begin{cases}
\displaystyle
\frac{\text{d}^2 x}{\text{d}t^2} = \delta \frac{\text{D}u}{\text{D}t} - \mu \left( \frac{\text{d} x}{\text{d}t} - u \right) \\ 
\displaystyle
\frac{\text{d}^2 y}{\text{d}t^2} = \delta \frac{\text{D}v}{\text{D}t} - \mu \left( \frac{\text{d} y}{\text{d}t} - v \right)
\end{cases}$$

with $$\delta=\frac{\rho_f}{\rho_p}$$ the ratio of the fluid density $\rho_f$ to the density $\rho_p$ of the advected particle, $\mu$ is a coefficient involved to quatify the Stokes drag. To begin, one can choose $\delta=1$ and $\mu=1$
