"""
   Collection of functions for __EDSPEC analysis__ of event series used
   in the scripts to prepare/reproduce the figures in the manuscript
   N. Marwan & T. Braun: _Power spectral estimate for discrete data_,
   Chaos, 2023
   
   (3rd March 2023)
"""


using FFTW
using Statistics
using Random, Distributions


"""
   editdistance(t1::Vector, t2::Vector, p0, p2) → edit distance metric
   
   Compute the __edit distance metric__ between two _binary_ sequences, specified
   by their time instances `t1` and `t2` using the cost parameters for 
   adding/deleting `p2` and for shifting `p0`.
"""
function editdistance(t1::Vector{Float64}, t2::Vector{Float64}, p0, p2)
    N1 = length(t1)
    N2 = length(t2)
    # initialize cost matrix
    G = zeros(N1 + 1, N2 + 1)

    G[:, 1] = 0:N1
    G[1, :] = 0:N2
    # iterate over segments
    if N1 > 0 && N2 > 0
        for (i, t1_) in enumerate(t1), (j, t2_) in enumerate(t2)
            try
                # cost components of different operations
                cost_del = G[i, j+1] + p2
                cost_add = G[i+1, j] + p2
                cost_shift = G[i, j] + p0 * abs(t1_ - t2_)
                G[i+1, j+1] = min(cost_del,cost_add,cost_shift)
            catch
                G[i+1, j+1] = min(G[i, j+1] + p2, G[i+1, j] + p2)
            end
        end
        # resulting distance matrix
        d = G[end,end]
        
    else
        d = abs(N1 - N2) * p2
    end
    return d
end



"""
   sorted_intersect(t1::Vector, t1::Vector, avgDist::Float)

   Find __overlapping sequence__ between two event series provided by
   their time instances `t1` and `t2`, allowing for some small
   time uncertainty `avgDist`.

   For examle, `t1` has smaller values than `t2`:\\
        e.g. `t1 = [ 3 5 9]; t2 = [4 8 12];`\\
        result should be `[5 9]` and `[4 8]` (for `avgDist = 0`)
"""
function sorted_intersect(t1::Vector{Float64}, t2::Vector{Float64}, avgDist::Float64)

    @assert issorted( t1 ) # ensure vectors are sorted
    @assert issorted( t2 )

    common_t1 = Vector{Float64}() # final resulting vector
    common_t2 = Vector{Float64}()
    
    # create/ cut first vector
    i = firstindex(t1)
    while i <= lastindex(t1)
        if t1[i] >= (t2[1] - avgDist)
            push!( common_t1, t1[i] )
            i += 1
        else
            i += 1
        end
    end
    
    # create/ cut second vector
    i = firstindex(t2)
    while i <= lastindex(t2)
        if t2[i] <= (t1[end] + avgDist)
            push!( common_t2, t2[i] )
            i += 1
        else
            i += 1
        end
    end
    
    return common_t1, common_t2
end



"""
   EDACF(t_events::Vector, lags::Vector, t_events2::Union{Vector, Nothing} = nothing, p2::Float = 1.0)
   
   Computes the __edit distance based auto-correlation function__ (EDACF) 
   for a series of event times `t_events` at the lags `lags`,
   and using the deletion cost parameter `p2`.
   
   Optionally, a second set of event times `t_events2` can be used 
   to compute the cross-correlation function.
"""
function EDACF(t_events::Vector{Float64}, lags::Vector{Float64}, t_events2::Union{Vector, Nothing}=nothing, p2::Float64=1.0)

    M = length(lags)
    C = zeros(length(lags)); # initiate the results matrix

    avgDist = abs(t_events[end] - t_events[1]) / length(t_events); # average distance between events

    # loop to get edit distance for different lags
    for (i, lag) in enumerate(lags)
        # shift time series
        t_events_shifted = t_events.+lag;
        
        # use only overlap, allowing small time before and after last point of time
        x, y = sorted_intersect(t_events, t_events_shifted, avgDist)
        
        num_events = max(length(x), length(y)); # max. number of events in overlapping region
        #@printf("%i\n", num_events)
        
        if num_events > 2 # stop if too few events
        
                t_start = min(x[1], y[1]); # start time of overlap sequence
                t_end = max(x[end], y[end]); # end time of overlap sequence
        
                p0 = num_events / abs(t_end - t_start); # parameter edit distance cost on shifting = inverse of average distance between events

                # maximal edit distance for normalisation
                maxEDx = editdistance(x, Vector{Float64}(), p0, p2);
                maxEDy = editdistance(y, Vector{Float64}(), p0, p2);
                maxED = max(maxEDx, maxEDy);
                
                # edit distance between time series and time shifted copy
                ED = editdistance(x, y, p0, p2);
                C[i] = 1 - (ED / maxED); # normalise and flip values to normalize
        else
                # remove last points in C vector and stop calculation
                n = length(C)
                deleteat!(C, i:n)
                deleteat!(lags, i:n)
                break
        end
    end

    return C
end


"""
   EDSPEC(t_events::Vector, lags::Vector, C::Union{Vector, Nothing} = nothing, p2::Float = 1.0)
   
   Computes the __edit distance based power spectrum estimate__ (EDSPEC) for 
   a series of event times `t_events`, using the lags in vector `lags`
   and the deletion cost parameter `p2`. The EDACF is automatically
   calculated within this function.
   
   Alternatively, if the EDACF is already available, the EDACF calculation
   can be skipped by providing the EDACF as input for the `C` variable.
"""
function EDSPEC(t_events::Vector{Float64}, lags::Vector{Float64}, C::Union{Vector, Nothing}=nothing, p2::Float64=1.0)
    Δt = diff(lags)[1] #sampling time

    # edit autocorrelation function:
    if isnothing(C)
        C = EDACF(t_events, lags)
    end

    # normalise EDACF:
    normC = (C .- median(C)) ./ std(C) 

    # fourier transform:
    F = fftshift(fft(normC))

    # estimate PSD:
    fs = 1.0/Δt
    pwr = abs.(fftshift(fft(normC)))
    freqs = fftshift(fftfreq(length(normC), fs))
    idx0 = Int64(ceil(length(pwr)/2)+1)

    return freqs[idx0:end], pwr[idx0:end]
end



"""
   EDSPEC_conf(t_events::Vector, lags::Vector, nsurr::Int, alph::Float, t_events2::Union{Vector, Nothing} = nothing, p2::Float = 1.0)
   
   Returns the __alpha-confidence level of EDSPEC__ based on `nsurr` random 
   event series, using the lags in vector `lags` and the deletion cost 
   parameter `p2`.
"""
function EDSPEC_conf(t_events::Vector{Float64}, lags::Vector{Float64}, nsurr::Int64, alph::Float64, t_events2::Union{Vector, Nothing}=nothing, p2::Float64=1.0)
    # second time series?
    if isnothing(t_events2)
        Nevents = length(t_events)
        tlast = t_events[end]
    else
        Nevents = length(t_events2)
        tlast = t_events2[end]
    end

    L = length(lags)
    Δt = diff(lags)[1]

    pwr = Array{Any}(undef,(nsurr,L));
    for j in 1:nsurr
        # draw as many random indices as there are events
        # final time point: last event (tlast)
        ridx = Int64.(sort(sample(1:tlast, Int64(Nevents), replace = false)))
        tev = float(collect(1:tlast))
        tsurr = tev[ridx]

        ## EDAC
        # edit autocorrelation function:
        C = EDACF(tsurr, lags, t_events2, p2)
        # normalise EDACF:
        normC = (C .- median(C)) ./ std(C) 

        ## EDSPEC
        # fourier transform:
        F = fftshift(fft(normC))
        # estimate PSD:
        pwr[j,:] = abs.(fftshift(fft(normC)))
    end
    
    # crop to right half of spectrum and generate correct frequency grid
    idx0 = Int64(ceil(L/2)+1)
    fs = 1.0/Δt
    freqs = fftshift(fftfreq(L, fs))[idx0:end]
    pwr_ = pwr[:,idx0:end]
    # Compute alpha-quantile over realizations
    SI = vec(mapslices(x -> quantile(x, alph), pwr_, dims = 1))

    return freqs, SI
end



"""
   findnearest(a::Vector, b::Float)

   Helper function to find the __nearest value__ in vector `a` to a 
   given reference value `b`.
"""
function findnearest(a::Vector{Float64}, b::Float64)
    return findmin(abs.(a .- b))[2];
end
