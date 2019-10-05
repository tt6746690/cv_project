function [scenes,shifts] = SceneNames(dataset)
% get a list of scene names for dataset {exp60,7patterns}
%
%
switch dataset
case "exp60"
    scenes = [
        "bowl"
        "buddha"
        "building"
        "candle"
        "cups"
        "flower"
        "jfk"
        "pens"
        "pillow"
        "shoe"
        "resolution"
        "chart"
    ]';
case "7patterns"
    scenes = [
        "chameleon"
        "cover"
        "cup"
        "giraffe"
        "head"
        "lamp"
        "minion"
        "sponge"
        "totem"
        "train"
    ]';

    % shift input image to match order of projector phase shifts
    shifts = [
        0,
        1,
        0,
        5,
        1,
        5,
        0,
        1,
        2,
        5,
    ];
case "alphabet"
    scenes = [
        "alphabet4"
        "alphabet5"
        "alphabet6"
        "alphabet7"
    ]';
otherwise
    warning(sprintf("did not have dataset %s",dataset));
end