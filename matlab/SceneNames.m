function scenes = SceneNames(dataset)
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
        "sponge"
        "lamp"
        "cup"
        "chameleon"
        "giraffe"
        "head"
        "minion"
        "train"
        "totem"
        "cover"
    ]';
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