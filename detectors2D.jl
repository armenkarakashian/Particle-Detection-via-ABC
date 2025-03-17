#!/usr/bin/env julia
###############################################################################
# 2D Time-Dependent Schrödinger Equation on [0,1]×[0,1]
# Still in progress!
###############################################################################
@info "Loading packages..."
using DifferentialEquations, LinearAlgebra, Plots, FFMPEG, Dates, Serialization

@info "Packages loaded."

# Set plotting backend for headless use
ENV["GKSwstype"] = "nul"
gr()

# Plot styling
Plots.default(size=(1920,1080))
title_font = font("Helvetica", 42)
other_font = font("Helvetica", 30)
default(
    titlefont  = title_font,
    guidefont  = other_font,
    tickfont   = other_font,
    legendfont = other_font
)

Nx = 128  # Number of interior points in x
Ny = 128  # Number of interior points in y

delta_x = 1/(Nx+1)
delta_y = 1/(Ny+1)

# PDE time parameters
tspan = (0.0, 5)
tN    = 1728   # Number of saved frames for animation

# Boundary parameter
k = 1.0

# Set up Gaussian wave packet as initial data
function initial_psi(Nx, Ny, delta_x, delta_y)
    psi = Matrix{ComplexF64}(undef, Nx+2, Ny+2)

    # Initialize grid
    fill!(psi, 0 + 0im)

    x0, y0 = 0.5, 0.5
    σ     = 0.1
    for j in 2:(Ny+1)
        yval = (j-1)*delta_y
        for i in 2:(Nx+1)
            xval = (i-1)*delta_x
            r2 = (xval - x0)^2 + (yval - y0)^2
            psi[i,j] = exp(-r2/(2σ^2))
        end
    end
    return psi
end

# Ghost cells (on the edges) are where we set the boundary conditions
function set_ghost_cells!(psi, Nx, Ny, delta_x, delta_y, k)
    # Left boundary: i = 1
    for j in 2:(Ny+1)
        psi[1, j] = (1 + im*k*delta_x) * psi[2, j]
    end

    # Right boundary: i = Nx+2
    for j in 2:(Ny+1)
        psi[Nx+2, j] = psi[Nx+1, j] + im*k*delta_x * psi[Nx+1, j]
    end

    # Bottom boundary: j = 1
    for i in 2:(Nx+1)
        psi[i, 1] = (1 + im*k*delta_y) * psi[i, 2]
    end

    # Top boundary: j = Ny+2
    for i in 2:(Nx+1)
        psi[i, Ny+2] = psi[i, Ny+1] + im*k*delta_y * psi[i, Ny+1]
    end

    # Corner handling: average the adjacent sides
    # Bottom left corner (i=1, j=1)
    psi[1, 1] = (psi[1, 2] + psi[2, 1]) / 2
    # Bottom right corner (i=Nx+2, j=1)
    psi[Nx+2, 1] = (psi[Nx+2, 2] + psi[Nx+1, 1]) / 2
    # Top left corner (i=1, j=Ny+2)
    psi[1, Ny+2] = (psi[1, Ny+1] + psi[2, Ny+2]) / 2
    # Top right corner (i=Nx+2, j=Ny+2)
    psi[Nx+2, Ny+2] = (psi[Nx+2, Ny+1] + psi[Nx+1, Ny+2]) / 2

    return
end

# ODE solver can only read 1D arrays
# The way I handle this function may be leading to the unexpected behavior
function flatten_psi(psi::Matrix{ComplexF64}, Nx, Ny)
    v = Vector{ComplexF64}(undef, Nx*Ny)
    idx = 1
    for j in 2:(Ny+1)
        for i in 2:(Nx+1)
            v[idx] = psi[i,j]
            idx += 1
        end
    end
    return v
end

function unflatten_psi(v::Vector{ComplexF64}, Nx, Ny)
    psi = Matrix{ComplexF64}(undef, Nx+2, Ny+2)
    fill!(psi, 0 + 0im)
    idx = 1
    for j in 2:(Ny+1)
        for i in 2:(Nx+1)
            psi[i,j] = v[idx]
            idx += 1
        end
    end
    return psi
end


function schrodinger2D!(dpsidt::Vector{ComplexF64}, v::Vector{ComplexF64}, p, t)
    Nx, Ny, delta_x, delta_y, k = p
    psi = unflatten_psi(v, Nx, Ny)
    # Update boundaries
    set_ghost_cells!(psi, Nx, Ny, delta_x, delta_y, k)
    
    # Compute dpsi on the interior
    dpsi_interior = Matrix{ComplexF64}(undef, Nx, Ny)
    for j in 2:(Ny+1)
        for i in 2:(Nx+1)
            lap = (psi[i+1, j] - 2*psi[i,j] + psi[i-1, j])/(delta_x^2) +
                  (psi[i, j+1] - 2*psi[i,j] + psi[i, j-1])/(delta_y^2)
            dpsi_interior[i-1,j-1] = im * lap
        end
    end
    
    # Flatten the result back into dpsidt
    idx = 1
    for j in 1:Ny
        for i in 1:Nx
            dpsidt[idx] = dpsi_interior[i,j]
            idx += 1
        end
    end
end

# Solve the system of ODEs
@info "Preparing to solve PDE..."
time_step_to_print = 0.05
next_time_trigger = Ref(tspan[1])
function condition(u, t, integrator)
    if t >= next_time_trigger[]
        next_time_trigger[] += time_step_to_print
        return true
    else
        return false
    end
end
function print_progress(integrator)
    fraction = integrator.t / integrator.sol.prob.tspan[end] 
    @info "PDE Progress: $(round(fraction * 100, digits=1))% (t = $(round(integrator.t, digits=2)))"
end
progress_cb = DiscreteCallback(condition, print_progress)

psi_init = initial_psi(Nx, Ny, delta_x, delta_y)
v0 = flatten_psi(psi_init, Nx, Ny)
p = (Nx, Ny, delta_x, delta_y, k)
prob = ODEProblem(schrodinger2D!, v0, tspan, p)

# Use TRBDF2 (Crank-Nicholson)
sol = solve(prob, TRBDF2(autodiff=false), saveat=range(tspan[1], tspan[2], length=tN), callback = progress_cb)

# Last step: animate and save
xvals = range(delta_x, step=delta_x, length=Nx)
yvals = range(delta_y, step=delta_y, length=Ny)

@info "Building animation of |psi|²..."
anim = @animate for (i, t_now) in enumerate(sol.t)
    psi_full = unflatten_psi(sol.u[i], Nx, Ny)

    density = zeros(Float64, Nx, Ny)
    for j in 2:(Ny+1)
        for i in 2:(Nx+1)
            density[i-1,j-1] = abs2(psi_full[i,j])
        end
    end
    # We use a heatmap for faster rendering
    p = heatmap(xvals, yvals, density', clim=(0, 1),
                  xlabel="x", ylabel="y",
                  title="t = $(round(t_now, digits=3)), |psi|²",
                  aspect_ratio=:equal, cbar=true)
    p
end

mp4(anim, "schrodinger_2D_boundary_condition.mp4", fps=1152)
@info "Animation saved: schrodinger_2D_boundary_condition.mp4"
