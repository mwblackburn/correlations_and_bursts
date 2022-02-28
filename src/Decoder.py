import copy

# Class that defines a decoder for a specific stimulus table.
# Right now, it stores everything remotely relevant to decoding.
# That can (and will) change as the package comes together more


class Decoder:

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

    def get_stimulus_type(self):
        return self.stimulus_type

    def unpack(self):
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
        if not self.has_weights():
            raise ValueError(f"{self.name} has no weights.")
        return (self.weights_by_bin, self.weights_by_modality, self.weights_by_cell)

    def add_weights(self, weights_by_bin, weights_by_modality, weights_by_cell) -> None:
        self.weights_by_bin = weights_by_bin
        self.weights_by_modality = weights_by_modality
        self.weights_by_cell = weights_by_cell
        return

    # To check whether this Decoder has been run
    def has_weights(self) -> bool:
        return not self.weights_by_bin is None
