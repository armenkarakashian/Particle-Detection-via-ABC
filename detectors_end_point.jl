# Load packages (takes 2-3 minutes)
println("Loading packages...")
flush(stdout)
using DifferentialEquations, CUDA, LinearAlgebra, Plots, FFMPEG, Dates, Serialization

# Necessary for animation creation without a monitor active (for some reason)
ENV["GKSwstype"] = "nul"
gr()

# Read command-line parameters
max_k1 = parse(Float32, get(ARGS, 1, "10.0"))
N = parse(Int, get(ARGS, 2, "50"))
L = 1.0f0
dx = L / (N - 1)

# Set up N spatial points
x = range(0f0, stop=L, length=N)

# Initial wave function: Gaussian wave packet
psi0 = exp.(-((x .- 0.5f0).^2) ./ (2 * 0.1f0^2)) .* ComplexF32(1, 0)
psi0 = cu(psi0)  # Prepare the initial condition for GPU processing

# Set up the Schrödinger model
function schrodinger(psi, p, t)
    k1 = p[1]
    k2 = k1 #0.5f0 * k1
    dpsi_dt = CUDA.zeros(ComplexF32, length(psi))
    dpsi_dt[2:N-1] .= im * (psi[3:N] - 2 .* psi[2:N-1] + psi[1:N-2]) / dx^2
    CUDA.@allowscalar begin
        dpsi_dt[1] = im * ((psi[1] / (1 - im * k1 * dx)) - 2 * psi[1] + psi[2]) / dx^2
        dpsi_dt[N] = im * (psi[N-1] - 2 * psi[N] + (psi[N] / (1 - im * k2 * dx))) / dx^2
    end
    return dpsi_dt
end

# Set size
default(size=(1920, 1080), dpi=300)

# Main plot initialization
plot(title="k*|ψ|^2 at x=1 for different k", xlabel="t", ylabel="k*|ψ|^2", legend=:topright)

# Solve the PDE for each k1 and plot
for k1 in 1.0f0:2:max_k1
    println("Simulating for k1 = k2 = $k1")
    flush(stdout)

    # Set up and solve the ODE problem on the GPU
    prob = ODEProblem(schrodinger, psi0, (0.0f0, 1.5f0), [k1])
    sol = solve(prob, TRBDF2(autodiff=false), saveat=0.0:0.0015:1.5)  # Store results at regular intervals

    println("Plotting for k1 = k2 = $k1")
    flush(stdout)
    # Extract and plot real part of ψ at x=1 over time
    CUDA.@allowscalar begin
        re_psi_at_x1 = [k1*abs2(sol.u[i][N]) for i in 1:length(sol.t)]  
    end

    plot!(sol.t, re_psi_at_x1, label="k = $k1")
end

# Save the combined plot to a file
savefig("max_k=$(max_k1)_k*abs2(Psi)_at_x1_over_time.png")
println("All simulations complete. Plot saved as 'Real_Psi_at_x1_over_time.png'.")

