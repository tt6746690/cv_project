function scenes = SceneNames(dataset)
% get a list of scene names for dataset {exp60,7pattern}
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
otherwise
    warning(sprintf("did not have dataset %s",dataset));
end