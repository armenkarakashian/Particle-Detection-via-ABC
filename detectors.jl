#
#	This setup uses Float32 for faster GPU computation
#

#	Load packages (takes 2-3 minutes)
println("Loading packages...")
flush(stdout)
using DifferentialEquations, CUDA, LinearAlgebra, Plots, FFMPEG, Dates, Serialization

#	Necessary for animation creation without a monitor active (for some reason)
ENV["GKSwstype"] = "nul"
gr()

#	Parameters
k1 = parse(Float32, get(ARGS, 1, "2.0"))
k2 = k1
#k2 = parse(Float32, get(ARGS, 2, "2.0"))
N = parse(Int, get(ARGS, 3, "50"))
tN = parse(Int, get(ARGS, 4, "200"))
L = 1.0f0
dx = L / (N - 1)

#	Set up N spatial points
x = range(0f0, stop=L, length=N)

#	Initial wave function: Gaussian wave packet
psi0 = exp.(-((x .- 0.5f0).^2) ./ (2 * 0.1f0^2)) .* ComplexF32(1, 0)

#	Alternative initial wave function for debugging
#
#	Constant amplitude across the entire domain
#A = 1.0f0 / sqrt(L)
#
#	Initial wave function: Constant across 0 to L
#psi0 = fill(ComplexF32(A, 0), N)
#

#	Set up model
function schrodinger(psi, p, t)
    dpsi_dt = CUDA.zeros(ComplexF32, length(psi))
    #	Interior points
    dpsi_dt[2:N-1] .= im * (psi[3:N] - 2 .* psi[2:N-1] + psi[1:N-2]) / dx^2
    #	Boundary conditions
    CUDA.@allowscalar begin
        dpsi_dt[1] = im * ((psi[1] / (1 - im * k1 * dx)) - 2 * psi[1] + psi[2])   / dx^2
        dpsi_dt[N] = im * (psi[N-1] - 2 * psi[N] + (psi[N] / (1 - im * k2 * dx))) / dx^2
        #dpsi_dt[1] = im * (psi[2] - psi[1]) / dx^2
        #dpsi_dt[N] = im * (psi[N-1] - psi[N]) / dx^2
    end
    return dpsi_dt
end

#	Ensure initial condition and Schrödinger function are ready to be placed on GPU
psi0 = cu(psi0)  

#	Set up the system of ODEs problem on the GPU
prob = ODEProblem(schrodinger, psi0, (0.0f0, 1.5f0))

#	Set up logging and estimated time til completion

#	Set up function to calculate mean
function mean(values::Vector{Float32})
    total = sum(values)
    count = length(values)
    return count > 0 ? total / count : 0f0
end

#	Time logging variables
start_time = now()
last_time = Ref(now())
times = Float32[]
next_time_trigger = Ref(0.0f0)
time_step = 0.01f0

#	Create function to convert seconds to days, hours, minutes, and seconds
function format_duration(seconds::Float64)
    # Create a Period from the total seconds
    total_seconds = round(Int, seconds)
    days = div(total_seconds, 86400)
    hours = div(total_seconds % 86400, 3600)
    minutes = div((total_seconds % 3600), 60)
    seconds = total_seconds % 60

    # Build a formatted string
    formatted_time = ""
    if days > 0
        formatted_time *= "$(days) day" * (days != 1 ? "s " : " ")
    end
    if hours > 0 || days > 0
        formatted_time *= "$(hours) hour" * (hours != 1 ? "s " : " ")
    end
    if minutes > 0 || hours > 0 || days > 0
        formatted_time *= "$(minutes) minute" * (minutes != 1 ? "s " : " ")
    end
    formatted_time *= "$(seconds) second" * (seconds != 1 ? "s" : "")

    return formatted_time
end

#	Create function for logging time and estimated completion
function time_logging_callback(integrator)
    current_time = now()
    # Calculate the interval in seconds properly
    interval_milliseconds = current_time - last_time[]  # This is a Millisecond object
    interval_seconds = interval_milliseconds.value / 1000  # Convert milliseconds to seconds
    push!(times, Float32(interval_seconds))  # Store as Float32 if needed
    last_time[] = current_time

    if length(times) > 1
        average_step_time = mean(times)
        elapsed_time = (current_time - start_time).value / 1000  # Convert to seconds
        total_time_span = integrator.sol.prob.tspan[2] - integrator.sol.prob.tspan[1]
        progress_fraction = (integrator.t - integrator.sol.prob.tspan[1]) / total_time_span
        total_estimated_time = elapsed_time / progress_fraction
        remaining_time = total_estimated_time - elapsed_time
        readable_time = format_duration(remaining_time)
        println("Current simulation time: $(integrator.t)/1.5, Done in: $(readable_time)")
    else
        println("Current simulation time: $(integrator.t), collecting data for estimation...")
    end
    
    flush(stdout)

    return nothing
end

#	Only update tN times
function callback_condition(u, t, integrator)
    if t >= next_time_trigger[]
        next_time_trigger[] += time_step
        return true
    end
    return false
end

#	Create callback
callback = DiscreteCallback(callback_condition, time_logging_callback, save_positions=(false, false))

#	Solve the system of ODEs
#	Use Crank-Nicolson method
sol = solve(prob, TRBDF2(autodiff=false), saveat=range(0.0f0, 1.5f0, length=tN), callback=callback)

#	Create directory to save results
folder_name = "k1=$(k1)_k2=$(k2)_N=$(N)_tN=$(tN)_$(Dates.now())"
mkdir(folder_name)

#	Save solution
serialize(joinpath(folder_name, "solution.jls"), sol)

println("Finished simulation, creating animations...")
flush(stdout)

#	Animate
function create_animation(x, sol, map_func, ylabel, title_str, filename)
    full_path = joinpath(folder_name, filename)

    anim = @animate for i in 1:length(sol.t)
        psi_data = Array(map_func(sol.u[i]))

        plot(x, psi_data, ylim=(0, 1),
             xlabel="x", ylabel=ylabel, title=title_str, legend=false)
    end

    mp4(anim, full_path, fps=100)
end

#	Generate animations for real, imaginary, and absolute square of the wave function
create_animation(x, sol, real, "Re(ψ(x, t))", "Evolution of Re(ψ) over time", "real_psi.mp4")
create_animation(x, sol, imag, "Im(ψ(x, t))", "Evolution of Im(ψ) over time", "imag_psi.mp4")
create_animation(x, sol, f -> abs.(f).^2, "|ψ(x, t)|²", "Evolution of |ψ|² over time", "abs2_psi.mp4")

println("Simulation and animations complete. Results saved in folder '$folder_name'.")

