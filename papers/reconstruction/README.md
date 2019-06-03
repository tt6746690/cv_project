

##### Structured light Q&A

+ phase shifting
    + algorithm for determining correspondence of the different phases
+ terminology 
    + spatial sinusoids
        + black stripes have intensity of 1 and white stripes have intensity of 0 and they corresponds to spatial sinusoids
    + frequency
        + 1 pattern with varying width column stripes 
    + shift N times
        + 1 pattern that is shifted by some N times, each by amount of phase
    + phase
        + each phase corresponds to a column stripe in a spatial sinusoids
+ where did the formula relating intensity and amplitude, offset, phase changes are ?
    + https://en.wikipedia.org/wiki/Phase_(waves)#Phase_shift
+ phase wrapping/unwrapping from interferometry paper (relation to phase shifting)
+ what is light transport coefficient a_{qc}
    + describes intensity of reflected light from point q that is received by point c?
+ why amplitude is a function of both frequency and amount of defocus?
+ why higher mean frequency -> resistance to global illumination
+ why frequency measured with respect to pixels
    + spatial frequency ?
    + w = 16 pixel period means how many 16 pixel in the image


+ [1991_automatic_processing_of_fringe_pattern_in_interger_interferometers](1991_automatic_processing_of_fringe_pattern_in_interger_interferometers.pdf)
    + original paper explaining micro-phasing theory
    + goal
        + determine total phase difference of light waves without interference fringe counting


+ [2003_pattern_codification_strategies_in_structured_light_system](2003_pattern_codification_strategies_in_structured_light_system.pdf)
    + coded structured light
        + for recovering surface of objects
        + no location constraint
        + idea
            + project a light pattern 
            + view the illuminated scene from one or more points of view
            + decoded points can be triangulated and 3D information obtained
        + patterns
            + the projected images are called patterns
    + stereo vision
        + view scene from two or more points of view and then finding correspondence between different images to triangulate the 3D position
    + coding strategies
        + patterns 
            + designed to have _codewords_ assigned to a _set of pixels_
            + f: codewords -> coordinates of pixel in pattern
                + codewords are numbers, i,e. grey levels, colors, 
        + based on what to classify
            + what kind of codeword is used
            + if codeword encodes a single axis or two spatial axis
        + time-multiplexing strategy
            + patterns successively projected onto measuring surface
            + codeword for a pixel: sequence of illumination values
            + the good
                + accurate because 
                    + codeword basis tends to be small, so distinguishable
                    + since pattern is successively projected, ...
            + subclasses
                + binary codes
                    + 0-1-1-0-1- ...for 1 pattern axis
                    + a sequence of m patterns, 
                        + 2^m  stripes, i.e. codewords, using a binary code
                        + codeword is a sequence of 0s and 1s from m patterns, first pattern is one contains most significant bit, 0-black, 1-white
                + n-ary 
                    + reduces number of patterns by means of increasing the number of intensity levels used to encode the stripes
                    + multilevel Gray code
                        + Gray code with alphabet of n symbols -> reduce number of patterns
                            + i.e. n^m instead of 2^m stripes
                + gray code combined with phase shifting
                    + same pattern projected several times, shifting it in certain direction in order to increase resolution
                        + Gray code methods unambiguous
                            + hamming distance of 1 -> good against noise
                        + phase shift methods
                            + high resolution
                + hybrid


+ [2012_micro_phase_shifting](2012_micro_phase_shifting.pdf)
    + http://www.cs.columbia.edu/CAVE/projects/micro_phase_shifting/
    + goal 
        + shape recovery addressing global allumination + illumination defocus
    + problems & solutions
        + global illumination   
            + project sinusoidal patterns with frequencies limited to a narrow frequency band, over which global illumination remain cosntant
        + defocus effects of projecter (limited FOV)
            + frequency in a narrow band, 
                + so amplitudes for all frequencies are approximated the same, ... single unknown
                + reduce number of input images for Micro PS
        + resolving depth ambiguities
            + fact
                + high freq sinusoids -> high resolution info, but in small depth range
                + use low freq sinusoids to disambiguate phase info over larger range
            + solution  
                + emulate a low frequency sinusoid with a period equal to product of periods of several high-frequency sinusoids
    + phase shifting
        + phase corresponds to correspondence of projector pixels -> depth ..
