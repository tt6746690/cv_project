


Nov 9 meeting with Mian/kyros

+ capture more pattern per frame
+ demosaicing algorithm

+ why is demosaicing algorithm bad?

+ what was tried before
    + linear interpolation from matlab `demosaic`
        + 1/N resolution 
        + interpolate linearly

+ approaches
    + cast existing demosaicing algorithm on tiles we have right now
        + adapt the best demosaicing algorithm and adapt to our problem
    + learn-based algorithm
        + given a large collection of scenes, try to learn the codes!
        + existing learning based algorithm
            + train from scratch

+ todo
    + literature search on state-of-the-arts algorithm
        + take note of how micro-phasing affect how the model is set up
        + shoud not expect the same kind of correlation from existing spatially varying algorithms
    + look for work that cite papers the _joint denoising_ or _assorted pixels_
        + finding data...
            + rendering in game engine?
            + synthesizing dataset without looking at 3D geometry ...
                + feasible, pipeline...
        + or take 2D image and multiply with pattern, as training data
            + but would not have depth information
    + identify top performing algorithms
        + non-learning based
            + probably have to implement them ourselves
        + re-train using data
    + paper reading
        + take look at both, performance on metrics
        + take look at insights on how they exploited correlation in natural images


+ endgoal
    + general demosaicing for micro-phasing, stereo-imaging
    + (mian) just determine the layout of mosaics/patterns, then use general purpose algorithms on them

+ tricky part 
    + at what level / how to model correlation for stereo-imaging