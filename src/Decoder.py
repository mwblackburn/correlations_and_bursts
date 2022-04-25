import copy

# Class that defines a decoder for a specific stimulus table.
# Right now, it stores everything remotely relevant to decoding.
# That can (and will) change as the package comes together more


class Decoder:
    """One line summary of Decoder
    
    More in depth description of Decoder
    
    Methods
    _______
    
    
    Attributes
    __________
    classifier : 
    
    stimulus_type : 
    
    stim_table : 
    
    stim_modalities : 
    
    bin_edges : 
    
    x : 
    
    y : 
    
    name : 
    
    weights_by_bin : 
    
    weights_by_modality : 
    
    weights_by_cell : 
    
    """

    # TODO: Doc me

    def __init__(
        self,
        classifier,
        stimulus_type,
        stim_table,
        stim_modalities,
        bin_edges,
        x,
        y,
        name="",
    ) -> None:
        """Create a new Decoder
        
        Note
        ____
        Maybe theres something to note?
        
        Parameters
        __________
        
        
        """
        self.classifier = classifier
        self.stimulus_type = stimulus_type
        self.stim_table = stim_table
        self.stim_modalities = stim_modalities
        self.bin_edges = bin_edges
        self.x = x
        self.y = y
        if name == "":
            name = copy.deepcopy(stimulus_type)
        self.name = name
        self.weights_by_bin = None
        self.weights_by_modality = None
        self.weights_by_cell = None
        self.accuracies_by_bin = None
        self.cv_accuracies = None
        self.whole_burst_dict = None
        self.whole_single_dict = None

    def get_stimulus_type(self):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        return self.stimulus_type

    def unpack(self):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        return (
            self.classifier,
            self.stimulus_type,
            self.stim_table,
            self.stim_modalities,
            self.bin_edges,
            self.x,
            self.y,
        )

    def unpack_weights(self):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        if not self.has_weights():
            raise ValueError(f"{self.name} has no weights.")
        return (self.weights_by_bin, self.weights_by_modality, self.weights_by_cell)

    def add_weights(
        self,
        weights_by_bin,
        weights_by_modality,
        weights_by_cell,
        accuracies_by_bin,
        cv_accuracies,
    ) -> None:
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        self.weights_by_bin = weights_by_bin
        self.weights_by_modality = weights_by_modality
        self.weights_by_cell = weights_by_cell
        self.accuracies_by_bin = accuracies_by_bin
        self.cv_accuracies = cv_accuracies
        return

    def add_bursts(self, whole_burst_dict, whole_single_dict):
        self.whole_burst_dict = whole_burst_dict
        self.whole_single_dict = whole_single_dict
        return

    # To check whether this Decoder has been run
    def has_weights(self) -> bool:
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        return self.weights_by_bin is not None

    def has_bursts(self) -> bool:
        return self.whole_burst_dict is not None

    def has_singles(self) -> bool:
        return self.whole_single_dict is not None
