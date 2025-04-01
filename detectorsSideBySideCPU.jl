@info "Loading packages..."
using DifferentialEquations, LinearAlgebra, Plots, FFMPEG, Dates, Serialization
using Plots.PlotMeasures

@info "Packages loaded, preparing to solve PDE(s)..."

ENV["GKSwstype"] = "nul"  #    For animation creation without an active monitor
gr()

#    Plot styling
Plots.default(size=(1920,1080))
title_font = font("Helvetica", 42)
other_font = font("Helvetica", 30)
default(
    titlefont  = title_font,
    guidefont  = other_font,
    tickfont   = other_font,
    legendfont = other_font
)

solutions = Dict{Float32, ODESolution}()
ks = []
global tN = nothing
global N = nothing

sol_dir = "solution_data"

#   Parameters
ks  = [0.5f0, 1.0f0, 2.0f0]  #    All values of k
global N   = 100                    #    Number of spatial points
global tN  = 200                   #    Number of frames
L   = 1.0f0
dx  = L / (N - 1)
x   = range(0f0, stop=L, length=N)

#    Initial wave function is Gaussian wave packet
psi0 = exp.(-((x .- 0.5f0).^2) ./ (2 * 0.1f0^2)) .* ComplexF32(1, 0)

#    Schrödinger equation ODE for a given k
function schrodinger!(dpsi_dt, psi, p, t)
    k1, k2, dx = p
    @inbounds begin
        #    Initialize the derivative vector with complex entries
        fill!(dpsi_dt, 0im)
    
        #    Derivative at interior points
        @inbounds for j in 2:length(psi)-1
            dpsi_dt[j] = im * (psi[j+1] - 2psi[j] + psi[j-1]) / dx^2
        end
    
        #    Absorbing boundary at x=0
        dpsi_dt[1] = im * ((psi[1] / (1 - im*k1*dx)) - 2*psi[1] + psi[2]) / dx^2
        #    Absorbing boundary at x=1
        dpsi_dt[end] = im * (psi[end-1] - 2*psi[end] + (psi[end] / (1 - im*k2*dx))) / dx^2
    end
end

#    Time span
tspan = (0.0f0, 1.5f0)

#    Progress callback (for printing)
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

#    Solve for each value of k
for k in ks
    @info "Solving PDE for k = $k ..."
    prob = ODEProblem(
        schrodinger!,
        psi0,
        tspan,
        (k, k, dx)
    )

    sol = solve(
        prob,
        TRBDF2(autodiff=false),
        saveat = range(tspan[1], tspan[2], length=tN),
        callback = progress_cb
    )

    solutions[k] = sol

    file_path = joinpath(sol_dir, "solution_k=$(k)_N=$(N)_tN=$(tN).jls")
    serialize(file_path, sol)
    println("Saved solution for k = $k to $file_path")

    @info "Finished solving for k = $k."
end

@info "All PDE solutions complete/loaded. Creating side-by-side animation..."

#   Frames
t_vals = Array(first(solutions).second.t)

#    For each solution, collect |psi|^2 for all times
abs2_at_x1 = Dict{Float32, Vector{Float32}}()

for k in ks
    sol = solutions[k]

    #    Evaluate |psi|^2 for each time at x = 1
    local_abs2_at_x1 = [abs2(Array(sol.u[i])[end]) for i in 1:length(sol.t)]

    abs2_at_x1[k] = local_abs2_at_x1
end

#      Y-limit for both plots (total probability is 1)
ymax = 1.1

anim = @animate for i in 1:length(t_vals)
    #    Left panel: multiple lines for different k
    p1 = plot(
        xlim=(0,1), ylim=(0,ymax),
        xlabel="x",
        title="|ψ(x,t)|² at t = $(round(t_vals[i], digits=3))",
        legend=:topright,
        left_margin=25px,
        linewidth=6
    )
    
    #    Add one series per k
    for k in ks
        sol = solutions[k]
        plot!(p1,
            x,
            abs2.(Array(sol.u[i])),
            label="k = $k"
        )
    end
    
    #    Right panel: multiple lines of |ψ(1,t)|² vs. t for different k
    p2 = plot(
        xlim=(0, t_vals[end]), ylim=(0,ymax),
        xlabel="t",
        title="k·|ψ(1,t)|² over time",
        legend=:topright,
        left_margin=25px,
        linewidth=6
    )
    
    for k in ks
        t_data = t_vals[1:i]
        y_data = k * abs2_at_x1[k][1:i]
        plot!(p2,
            t_data,
            y_data,
            label="k = $k"
        )
    end
    
    plot(p1, p2, layout=(1,2))
    
    fraction = i / length(t_vals)
    @info "Animation Progress: $(round(fraction * 100, digits=1))% (frame = $i of $(length(t_vals)))"
end

#    Save animation
mp4(anim, "compare_ks.mp4", fps=1152)
@info "Animation complete! Saved as 'compare_ks.mp4'."
