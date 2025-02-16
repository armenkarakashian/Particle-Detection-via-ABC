@info "Loading packages..."
using DifferentialEquations, LinearAlgebra, Plots, FFMPEG, Dates, Serialization
using Plots.PlotMeasures

@info "Packages loaded, solving PDE..."

ENV["GKSwstype"] = "nul" #   For animation creation without an active monitor (for job scheduling, other users can comment this out if desired)
gr()

Plots.default(
    size = (3840, 2160)
)
title_font = font("Helvetica", 42)  # Replace 18 with your desired font size
other_font = font("Helvetica", 30)
default(titlefont=title_font, guidefont=other_font, tickfont=other_font, legendfont=other_font)
#=
# FOR LOADING EXISTING SOLUTION TO CREATE ANIMATION

sol = deserialize(joinpath("solution_data", "side_by_side_sol.jls"))
t_vals = Array(sol.t)
abs2_at_x1 = [abs2(Array(sol.u[i])[end]) for i in 1:length(sol.t)]
max_abs2_left  = maximum([maximum(abs2.(Array(u))) for u in sol.u])
max_abs2_right = maximum(abs2_at_x1)
ymax_left  = 1.1 * max_abs2_left
ymax_right = 1.1 * max_abs2_right

anim = @animate for i in 1:length(t_vals)
    p1 = plot(x,
              abs2.(Array(sol.u[i])),
              xlabel="x", ylabel="|ψ(x,t)|²",
              title="|ψ(x,t)|² at t = $(round(t_vals[i], digits=3))",
              legend=false, ylim=(0, ymax_left), left_margin=100)
    
    y_data = abs2_at_x1[1:i]
    t_data = t_vals[1:i]
    p2 = plot(t_data,
              y_data,
              xlabel="t", ylabel="|ψ(1,t)|²",
              title="Value at x=1 over time",
              legend=false, xlim=(0, t_vals[end]), ylim=(0, ymax_right), left_margin=100)
    
    x_curr = t_data[end]
    y_curr = y_data[end]
    plot!(p2, [0, x_curr], [y_curr, y_curr], linewidth=1)
    plot!(p2, [x_curr, x_curr], [0, y_curr], linewidth=1)
    
    plot(p1, p2, layout=(1,2))
end

mp4(anim, "loaded_solution_animation.mp4", fps=30)
println("Animation from loaded solution complete!")
=#

# Parameters
k1 = 1.0f0     #    Set absorbing boundary condition constant to 1 for simplicity
k2 = k1        
N  = 864        #    number of points to break down [0,1] space interval
tN = 1728       #    animation frames. 1.5 seconds, 144 frames/s, intended to slow down to 0.125x speed
L  = 1.0f0
dx = L / (N - 1)

#   Break up [0,1] into subintervals
x = range(0f0, stop=L, length=N)

#   Gaussian wave packet as initial wave function
psi0 = exp.(-((x .- 0.5f0).^2) ./ (2 * 0.1f0^2)) .* ComplexF32(1, 0)

#   Schrödinger equation ODE
function schrodinger(psi, p, t)
    #   Initialize data structure to store derivative at every point
    dpsi_dt = zeros(ComplexF32, length(psi))

    #   Approximate derivative for interior points
    dpsi_dt[2:N-1] .= im * (psi[3:N] - 2 .* psi[2:N-1] + psi[1:N-2]) / dx^2

    #   Derivative for endpoints (using ABC)
    dpsi_dt[1] = im * ((psi[1] / (1 - im * k1 * dx)) - 2*psi[1] + psi[2])   / dx^2
    dpsi_dt[N] = im * ( psi[N-1] - 2*psi[N] + (psi[N] / (1 - im * k2 * dx)) ) / dx^2

    return dpsi_dt
end

tspan = (0.0f0, 1.5f0)

#   Define a callback condition that triggers every 0.05 units of time
#   and a function that prints progress
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

# Create a DiscreteCallback that uses the condition above
progress_cb = DiscreteCallback(condition, print_progress)

#   Set up and solve the ODE problem (t from 0 to tL=1.5)
prob = ODEProblem(schrodinger, psi0, (0.0f0, 1.5f0))
sol  = solve(prob, TRBDF2(autodiff=false), saveat=range(0.0f0, 1.5f0, length=tN), callback=progress_cb)

@info "Finished solving for k=1. Creating side-by-side animation..."

#   Save the solution
folder_name = "solution_data"
isdir(folder_name) || mkdir(folder_name)
serialize(joinpath(folder_name, "side_by_side_sol_N=$(N)_tN=$(tN).jls"), sol)
@info "Solution saved as ", joinpath(folder_name, "side_by_side_sol_N=$(N)_tN=$(tN).jls")


#   Put t vals onto CPU
t_vals = Array(sol.t)

#   Collect the values of |psi(1,t)|^2 over all times (and separately at x = 1)
abs2_at_x1 = [abs2(Array(sol.u[i])[end]) for i in 1:length(sol.t)]

#   Set the height of the graph based on maximum values attained by abs2
max_abs2_left  = maximum([maximum(abs2.(Array(u))) for u in sol.u])
max_abs2_right = maximum(abs2_at_x1)
ymax = max(max_abs2_left, max_abs2_right) * 1.1

#   Animation processing starts here
anim = @animate for i in 1:length(t_vals)
    #   Left animation: evolution of |psi|^2 on [0,1] over time
    p1 = plot(x,
              abs2.(Array(sol.u[i])),
              xlabel="x", ylabel="|ψ(x,t)|²",
              title = "|ψ|² at t = $(round(t_vals[i], digits=3))",
              legend=false,
              ylim=(0, ymax),
              left_margin=25px,
              linewidth=6)
    
    #   Right animation: trace of |psi|^2 at x = 1 over time
    #   Plot from t = 1 to current time t_vals[i]
    y_data = abs2_at_x1[1:i]
    t_data = t_vals[1:i]
    p2 = plot(t_data,
              y_data,
              xlabel="t", ylabel="",
              title="|ψ|² at x=1",
              legend=false,
              xlim=(0, t_vals[end]),
              ylim=(0, ymax),
              left_margin=25px,
              linewidth=6)

    x_curr = t_data[end]
    y_curr = y_data[end]
    plot!(p2, [0, x_curr], [y_curr, y_curr], linewidth=3)
    plot!(p2, [x_curr, x_curr], [0, y_curr], linewidth=3)
    
    plot(p1, p2, layout=(1,2))

    fraction = i / length(t_vals)
    @info "Animation Progress: $(round(fraction * 100, digits=1))% (t = $(i))"
end

#   Save completed animation!
mp4(anim, "side_by_side_N=$(N)_tN=$(tN).mp4", fps=1152)#576)

@info "Animation complete! Saved as 'side_by_side_N=$(N)_tN=$(tN).mp4'."
