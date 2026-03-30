module QSM

# Polyester.jl's @batch uses cfunction closures which are unsupported on
# ARM64 in Julia ≥1.12. Detect at compile time and provide a fallback.
const _USE_POLYESTER = !(Sys.ARCH === :aarch64 && VERSION >= v"1.12")

using Base: @propagate_inbounds, require_one_based_indexing
using Base.Threads: nthreads

using CPUSummary: num_cores
using DSP: Windows
using FastPow: @fastpow
using IrrationalConstants: inv2π, twoπ
using LinearMaps: LinearMap
using MacroTools: @capture, postwalk
using NIfTI: NIVolume, niread, niwrite
@static if _USE_POLYESTER
    using PolyesterWeave: reset_workers!
end
using Printf: @printf
using SLEEFPirates: pow, sincos_fast
using Static: known
using StaticArrays: SVector
@static if _USE_POLYESTER
    using ThreadingUtilities: initialize_task
end
using TiledIteration: TileIterator, padded_tilesize

using LinearAlgebra
using FFTW

@static if _USE_POLYESTER
    using Polyester
else
    @warn "Polyester @batch disabled (ARM64 + Julia ≥1.12 cfunction limitation), using Threads.@threads fallback"
    macro batch(args...)
        loop = args[end]
        # Find threadlocal=init::T keyword — extract the init expression
        tl_init = nothing
        for a in args[1:end-1]
            if a isa Expr && a.head == :(=) && a.args[1] == :threadlocal
                # a.args[2] is either `init::T` or just `init`
                init_expr = a.args[2]
                if init_expr isa Expr && init_expr.head == :(::)
                    tl_init = init_expr.args[1]  # the init value, drop type assert
                else
                    tl_init = init_expr
                end
            end
        end
        if tl_init !== nothing
            # Run serially with threadlocal as a regular local variable
            return esc(quote
                let threadlocal = $tl_init
                    $loop
                end
            end)
        else
            return esc(:(Threads.@threads $loop))
        end
    end
end


export bet
export gradfp, gradfp!, gradfp_adj, gradfp_adj!, lap, lap!
export dipole_kernel, laplace_kernel, smv_kernel
export homodyne, homodyne!, makewindow
export multi_echo_average, multi_echo_average!
export multi_echo_linear_fit, multi_echo_linear_fit!
export r2star_arlo, r2star_crsi, r2star_ll, r2star_numart2s
export r2star_arlo!, r2star_crsi!, r2star_ll!, r2star_numart2s!
export crop_mask, crop_indices, erode_mask, erode_mask!
export fastfftsize, padfastfft, padarray!, unpadarray, unpadarray!, psf2otf
include("utils/utils.jl")

export unwrap_laplacian
include("unwrap/unwrap.jl")

export ismv, lbv, pdf, sharp, vsharp
include("bgremove/bgremove.jl")

export nltv, rts, tikh, tkd, tsvd, tv
include("inversion/inversion.jl")


function __init__()
    @static if FFTW.fftw_provider == "fftw"
        fftw_set_threading(:FFTW)
    end
    return nothing
end


#####
##### Polyester.jl
#####

function reset_threading()
    # if @batch loop gets interrupted, threading has to be reset:
    # https://github.com/JuliaSIMD/Polyester.jl/issues/30
    if _POLYESTER_OK
        nt = min(nthreads(), (Sys.CPU_THREADS)::Int) - 1
        reset_workers!()
        foreach(initialize_task, 1:nt)
    end
    return nothing
end


#####
##### FFTW.jl
#####

const FFTW_NTHREADS = Ref{Int}(known(num_cores()))

@static if FFTW.fftw_provider == "fftw"
    # modified `FFTW.spawnloop` to use Polyester for multi-threading
    # https://github.com/JuliaMath/FFTW.jl/blob/v1.4.5/src/providers.jl#L49
    function _fftw_spawnloop(f::Ptr{Cvoid}, fdata::Ptr{Cvoid}, elsize::Csize_t, num::Cint, ::Ptr{Cvoid})
        @batch for i in 0:num-1
            ccall(f, Ptr{Cvoid}, (Ptr{Cvoid},), fdata + elsize*i)
        end
        return nothing
    end

    function fftw_set_threading(lib::Symbol = :FFTW)
        checkopts(lib, (:FFTW, :Polyester, :Threads), :lib)

        if lib ∈ (:Polyester, :Threads) && nthreads() < 2
            @warn "Cannot use $lib with FFTW. Defaulting to FFTW multi-threading" Threads.nthreads()
            lib = :FFTW
        end

        if lib == :Polyester
            cspawnloop = @cfunction(
                _fftw_spawnloop,
                Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid})
            )
        elseif lib == :Threads
            cspawnloop = @cfunction(
                FFTW.spawnloop,
                Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid})
            )
        else
            cspawnloop = C_NULL
        end

        # FFTW.jl ≥1.10 (Julia 1.12+) wraps library handles in
        # FakeLazyLibrary whose .h field is populated lazily on first
        # FFT call. Trigger both Float64 and Float32 library loading,
        # then resolve the threading callback via dlsym.
        function _fftw_handle(lib)
            if hasproperty(lib, :h)
                return lib.h
            else
                return lib[]
            end
        end

        # Ensure both double and single precision libraries are loaded
        h3  = _fftw_handle(FFTW.libfftw3)
        if h3 == C_NULL
            fft(Float64[1.0])  # trigger double-precision library load
            h3 = _fftw_handle(FFTW.libfftw3)
        end
        h3f = _fftw_handle(FFTW.libfftw3f)
        if h3f == C_NULL
            fft(Float32[1.0f0])  # trigger single-precision library load
            h3f = _fftw_handle(FFTW.libfftw3f)
        end

        fptr3  = ccall(:dlsym, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), h3,  "fftw_threads_set_callback")
        fptr3f = ccall(:dlsym, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), h3f, "fftwf_threads_set_callback")

        ccall(fptr3,  Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), cspawnloop, C_NULL)
        ccall(fptr3f, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), cspawnloop, C_NULL)

        return nothing
    end
end


end # module
