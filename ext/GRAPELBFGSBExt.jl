# SPDX-FileCopyrightText: © 2025 Michael Goerz <mail@michaelgoerz.net>
#
# SPDX-License-Identifier: MIT

module GRAPELBFGSBExt

import LBFGSB
using GRAPE: GrapeWrk, update_result!
import GRAPE: run_optimizer, gradient, step_width, search_direction, norm_search


function run_optimizer(optimizer::LBFGSB.L_BFGS_B, wrk, fg!, callback, check_convergence!)

    m = get(wrk.kwargs, :lbfgsb_m, 10)
    factr = get(wrk.kwargs, :lbfgsb_factr, 1e1)
    # LBFGSB stops when the relative reduction in the functional f is
    #     (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch
    pgtol = get(wrk.kwargs, :lbfgsb_pgtol, 1e-15)
    # LBFGSB will stop when the i'th component of the projected gradient g_i is
    #     max{|proj g_i | i = 1, ..., n} <= pgtol
    # We set both `factr` and `pgtol` to "extreme" high accuracy. We really
    # want GRAPE to control the convergence check, not LBFGSB.
    iprint = get(wrk.kwargs, :lbfgsb_iprint, -1)
    trace_debugging = (iprint == 100)
    x = wrk.pulsevals
    n = length(x)
    obj = optimizer
    f = 0.0
    # clean up
    fill!(obj.task, Cuchar(' '))
    fill!(obj.csave, Cuchar(' '))
    fill!(obj.lsave, zero(Cint))
    fill!(obj.isave, zero(Cint))
    fill!(obj.dsave, zero(Cdouble))
    fill!(obj.wa, zero(Cdouble))
    fill!(obj.iwa, zero(Cint))
    fill!(obj.g, zero(Cdouble))
    fill!(obj.nbd, zero(Cint))
    fill!(obj.l, zero(Cdouble))
    fill!(obj.u, zero(Cdouble))
    # set bounds
    for i = 1:n
        obj.nbd[i] = 0
        obj.l[i] = 0.0
        obj.u[i] = 0.0
        if wrk.lower_bounds[i] > -Inf
            obj.nbd[i] += 1
            obj.l[i] = wrk.lower_bounds[i]
        end
        if wrk.upper_bounds[i] > -Inf
            if obj.nbd[i] == 1
                obj.nbd[i] = 2  # lower and upper bound
            else
                obj.nbd[i] = 3  # upper bound only
            end
            obj.u[i] = wrk.upper_bounds[i]
        end
    end
    # start
    obj.task[1:5] = b"START"
    message_in = "START"
    message_out = ""

    while true # task loop

        message_in = strip(String(copy(obj.task)))
        # println("- start of task loop: $message_in")
        # println("Calling setulb with f=$f, |g|=$(norm(obj.g)), |x|=$(norm(x))")
        LBFGSB.setulb(
            n,
            m,
            x,
            obj.l,
            obj.u,
            obj.nbd,
            f,
            obj.g,
            factr,
            pgtol,
            obj.wa,
            obj.iwa,
            obj.task,
            iprint,
            obj.csave,
            obj.lsave,
            obj.isave,
            obj.dsave
        )
        message_out = strip(String(copy(obj.task)))
        # println("  task -> $message_out")

        if obj.task[1:2] == b"FG" # FG_LNSRCH or FG_START
            f = fg!(f, obj.g, x)
            # println("calling fg! for |x|=$(norm(x)) -> f=$f, |g|=$(norm(obj.g))")
            if obj.task[1:5] == b"FG_ST" # FG_START
                # x is the guess for the 0 iteration
                copyto!(wrk.gradient, obj.g)
                update_result!(wrk, 0)
                info_tuple = callback(wrk, 0)
                wrk.fg_count .= 0
                if !(isnothing(info_tuple) || isempty(info_tuple))
                    push!(wrk.result.records, info_tuple)
                end
            end
        elseif obj.task[1:5] == b"NEW_X"
            # x is the optimized pulses for the current iteration
            iter = wrk.result.iter + 1  # Cf. `obj.isave[30]`
            update_result!(wrk, iter)
            info_tuple = callback(wrk, wrk.result.iter)
            wrk.fg_count .= 0
            if !(isnothing(info_tuple) || isempty(info_tuple))
                push!(wrk.result.records, info_tuple)
            end
            check_convergence!(wrk.result)
            if wrk.result.converged
                fill!(obj.task, Cuchar(' '))
                obj.task[1:24] = b"STOP: NEW_X -> CONVERGED"
            else # prepare for next iteration
                copyto!(wrk.pulsevals_guess, x)
                copyto!(wrk.gradient, obj.g)
            end
        else
            if wrk.result.message == "in progress"
                wrk.result.message = String(copy(obj.task))
            end
            fill!(obj.task, Cuchar(' '))
            obj.task[1:10] = b"STOP: DONE"
            if trace_debugging
                print_lbfgsb_trace(wrk, obj, message_in, message_out)
            end
            break
        end
        if trace_debugging
            print_lbfgsb_trace(wrk, obj, message_in, message_out)  # enable for trace-debugging
        end

    end # task loop

    return nothing

end


function print_lbfgsb_trace(
    wrk,
    optimizer::LBFGSB.L_BFGS_B,
    message_in::AbstractString,
    message_out::AbstractString;
    show_details=true
)
    n = length(wrk.pulsevals)
    println("- end of task loop: $message_in -> $message_out")
    #! format: off
    if occursin("NEW_X", message_out) && show_details
        println("   lsave[1] = $(optimizer.lsave[1]):\t initial x has been replaced by its projection in the feasible set?")
        println("   lsave[2] = $(optimizer.lsave[2]):\t problem is constrained?")
        println("   lsave[3] = $(optimizer.lsave[3]):\t every variable is upper and lower bounds?")
        println("   isave[22] = $(optimizer.isave[22]):\t total number of intervals explored in the search of Cauchy points")
        println("   isave[26] = $(optimizer.isave[26]):\t the total number of skipped BFGS updates before the current iteration")
        println("   isave[30] = $(optimizer.isave[30]):\t the number of current iteration")
        println("   isave[31] = $(optimizer.isave[31]):\t the total number of BFGS updates prior the current iteration")
        println("   isave[33] = $(optimizer.isave[33]):\t the number of intervals explored in the search of Cauchy point in the current iteration")
        println("   isave[34] = $(optimizer.isave[34]):\t the total number of function and gradient evaluations")
        println("   isave[36] = $(optimizer.isave[36]):\t the number of function value or gradient evaluations in the current iteration")
        println("   isave[37] = $(optimizer.isave[37]):\t subspace argmin is (0) within / (1) beyond the box")
        println("   isave[38] = $(optimizer.isave[38]):\t the number of free variables in the current iteration")
        println("   isave[39] = $(optimizer.isave[39]):\t the number of active constraints in the current iteration")
        println("   isave[40] = $(optimizer.isave[40]):\t n+1-isave[40] = $(n+1-optimizer.isave[40]) = the number of variables leaving the set of active constraints in the current iteration")
        println("   isave[41] = $(optimizer.isave[41]):\t the number of variables entering the set of active constraints in the current iteration")
        println("   dsave[01] = $(optimizer.dsave[1]):\t current θ in the BFGS matrix")
        println("   dsave[02] = $(optimizer.dsave[2]):\t f(x) in the previous iteration")
        println("   dsave[03] = $(optimizer.dsave[3]):\t factr*epsmch")
        println("   dsave[04] = $(optimizer.dsave[4]):\t 2-norm of the line search direction vector")
        println("   dsave[05] = $(optimizer.dsave[5]):\t the machine precision epsmch generated by the code")
        println("   dsave[07] = $(optimizer.dsave[7]):\t the accumulated time spent on searching for Cauchy points")
        println("   dsave[08] = $(optimizer.dsave[8]):\t the accumulated time spent on subspace minimization")
        println("   dsave[09] = $(optimizer.dsave[9]):\t the accumulated time spent on line search")
        println("   dsave[11] = $(optimizer.dsave[11]):\t the slope of the line search function at the current point of line search")
        println("   dsave[12] = $(optimizer.dsave[12]):\t the maximum relative step length imposed in line search")
        println("   dsave[13] = $(optimizer.dsave[13]):\t the infinity norm of the projected gradient")
        println("   dsave[14] = $(optimizer.dsave[14]):\t the relative step length in the line search")
        println("   dsave[15] = $(optimizer.dsave[15]):\t the slope of the line search function at the starting point of the line search")
        println("   dsave[16] = $(optimizer.dsave[16]):\t the square of the 2-norm of the line search direction vector")
    end
    #! format: on
end


function gradient(wrk::GrapeWrk{O}; which=:initial) where {O<:LBFGSB.L_BFGS_B}
    if which == :initial
        return wrk.gradient
    elseif which == :final
        return wrk.optimizer.g
    else
        throw(ArgumentError("`which` must be :initial or :final, not $(repr(which))"))
    end
end

function step_width(wrk::GrapeWrk{O}) where {O<:LBFGSB.L_BFGS_B}
    return wrk.optimizer.dsave[14]
end


function search_direction(wrk::GrapeWrk{O}) where {O<:LBFGSB.L_BFGS_B}
    n = length(wrk.pulsevals)
    n0 = wrk.optimizer.isave[13]
    return wrk.optimizer.wa[n0:(n0+n-1)]
end


function norm_search(wrk::GrapeWrk{O}) where {O<:LBFGSB.L_BFGS_B}
    n = length(wrk.pulsevals)
    n0 = wrk.optimizer.isave[13]
    r = 0.0
    for i = n0:(n0+n-1)
        r += wrk.optimizer.wa[i]^2
    end
    return sqrt(r)
end

end
