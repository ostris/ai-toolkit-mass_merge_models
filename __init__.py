# This is an example extension for custom training. It is great for experimenting with new ideas.
from toolkit.extension import Extension


# We make a subclass of Extension
class ExampleMergeExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "mass_merge_models"

    # name is the name of the extension for printing
    name = "Mass Merge Models"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .MassMergeModels import MassMergeModels
        return MassMergeModels

# We make a subclass of Extension
class MassMergeCSI(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "mass_merge_models_csi"

    # name is the name of the extension for printing
    name = "Mass Merge Models CSI"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .MassMergeModelsCSI import MassMergeModelsCSI
        return MassMergeModelsCSI


AI_TOOLKIT_EXTENSIONS = [
    # you can put a list of extensions here
    ExampleMergeExtension,
    MassMergeCSI
]
