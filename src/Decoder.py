import copy
# Class that defines a decoder for a specific stimulus table.
# Right now, it stores everything remotely relevant to decoding.
# That can (and will) change as the package comes together more

class Decoder:
    
    # TODO: Doc me
    
    def __init__(self, classifier, stimulus_type, stim_table, bin_edges, x, y, name="") -> None:
        self.classifier = classifier
        self.stimulus_type = stimulus_type
        self.stim_table = stim_table
        self.bin_edges = bin_edges
        self.x = x
        self.y = y
        if name == "": name = copy.deepcopy(stimulus_type)
        self.name = name
        
    def get_stimulus_type(self):
        return self.stimulus_type