import numpy as np
from sklearn.svm import LinearSVC
from src.Decoder import Decoder

# import allensdk.brain_observatory.ecephys.ecephys_session

# This is where most of the calculations on a single session are going to be performed.
# It will:
# construct and evaluate decoders
# construct and evaluate PSTHs
# construct and evaluate correlation statistics
# save any relevant data for plots (so basically everything)


class SessionProcessor:

    # TODO: Doc me

    def __init__(self, session):
        # Set the internal session and id number
        self.session = session
        self._session_id = self.session.ecephys_session_id

        # Organize the session units by location
        # Key: acronym, Val: every unit in that area
        self.units_by_acronym = {}

        # Key: acronym, Val: tuple of indicies (start, stop)
        # start: the index of the first cell in self.all_units belonging to acronym
        # stop: the index of the first cell in self.all_units belonging to the next acronym
        self._all_unit_dividing_indices = {}

        # All the units in one convenient spot
        self.all_units = []
        for acronym in session.structure_acronyms:
            start = len(self.all_units)
            current = list(
                session.units[session.units.ecephys_structure_acronym == acronym].index
            )

            self.units_by_acronym[acronym] = current
            self.all_units = self.all_units + current
            stop = len(self.all_units)
            self._all_unit_dividing_indices[acronym] = (start, stop)

        # Eventually this Processor is going to be decoding stimuli, so we'll need to store
        # the decoders
        self._decoders = {}
        self._histograms = {}
        self._modality_histograms = {}

        # TODO: Construct more variables as needed

    # In theory, you could cause a bug here by calling this twice, with the same stim and bin
    # bin width, but different start and stop times. Right now, this issue is addressed with
    # an exception (ValueError)
    def construct_decoder(
        self,
        stimulus_type,
        name="",
        bin_start=0,
        bin_stop=0,
        bin_width=0.05,
        classifier=LinearSVC(),
    ):
        # If a name wasn't provided, name the decoder as explicitly as possible
        if name == "":
            name = f"{stimulus_type}_width_{bin_width*1000}ms"
        
        if name in self._decoders.keys():
            raise ValueError(
                f"A decoder with the name {name}, already exists. This can happen if you have constructed a decoder with the same stimulus type and bin width."
            )

        # Get the stim info
        stim_table = self.session.get_stimulus_table(stimulus_type)
        stim_ids = stim_table.index.values

        # Create the time bins
        bins = self._make_bins(bin_start, bin_stop, bin_width, stim_table)

        # Collect the data
        x = np.array(
            self.session.presentationwise_spike_counts(bins, stim_ids, self.all_units)
        )
        y = np.array(stim_table[stimulus_type])

        # Change the null labels to a numerical value, so that the classifiers don't panic
        for idx in range(len(y)):
            if y[idx] == "null":
                y[idx] = -1.0
        stim_modalities = np.unique(y)

        # Construct the decoder
        self._decoders[name] = Decoder(
            classifier,
            stimulus_type,
            stim_table,
            stim_modalities,
            bins,
            x,
            y,
            name=name,
        )
        return  # TODO: This function may not be complete

    # Idea: Eventually what we want, is to pass a set of stim names, and return psths, weights,
    # correlations, etc for every stim

    # Must be called after construct_decoder is called
    def construct_psth(self, name):
        if not name in self._decoders.keys():
            raise ValueError(
                f"{name} did not match the name of any decoders constructed by this object. You must call SessionProcessor.construct_decoders(args) before constructing PSTHs."
            )
        else:
            (
                classifier,
                stimulus_type,
                stim_table,
                stim_modalities,
                bins,
                x,
                y,
            ) = self._decoders[name].unpack()
            stim_presentation_ids = stim_table.index.values

        self._histograms[name] = self.session.presentationwise_spike_counts(
            bins, stim_table.index.values, self.all_units
        )
        modality_indicies = self._sort_by_modality(
            stim_modalities, y, stim_presentation_ids
        )

        modality_histograms = {}
        for stim in stim_modalities:
            modality_histograms[stim] = self.session.presentationwise_spike_counts(
                bins, modality_indicies[stim], self.all_units
            )
        self._modality_histograms[name] = modality_histograms
        return  # TODO: This function may not be complete

    def calculate_decoder_weights(self, name):
        # Check that the decoder exists, unpack it's information
        if not name in self._decoders.keys():
            raise ValueError(
                f"{name} did not match the name of any decoders constructed by this object."
            )
        else:
            (
                classifier,
                stimulus_type,
                stim_table,
                stim_modalities,
                bins,
                x,
                y,
            ) = self._decoders[name].unpack()
            stim_presentation_ids = stim_table.index.values

        # Get data information
        num_presentations, num_bins, num_units = x.shape
        yTrue = y.astype(int)

        # Initialize everything
        weights_by_modality = self._initialize_dict(
            stim_modalities, (num_bins, num_units)
        )
        weights_by_bin = {}
        weights_by_cell = {}

        # Train the classifier by bin, then store the resulting weights
        for bin in range(num_bins):
            # Get the data for the current time bin
            xBin = x[:, bin, :]

            # Train the classifier
            classifier.fit(xBin, yTrue)

            # Store the weights, and the classes.
            # The classes must be stored so that the correct set of
            # weights can be associated with the correct stim
            bin_weights = classifier.coef_
            classes = classifier.classes_
            weights_by_bin[bin] = bin_weights

            # Store the weights, sorted by modality
            idx = 0
            for stim in classes:
                weights_by_modality[stim][bin, :] = bin_weights[idx, :]
                idx += 1

        # Sort the weights by unit
        unit_idx = 0
        for unit_id in self.all_units:
            modality_idx = 0
            for stim in stim_modalities:
                # The modality_idx^th column of that particular cell, which should be every time bin
                # for the stimulus modality indicated by the stim
                weights_by_cell[unit_id][:, modality_idx] = weights_by_modality[stim][
                    :, unit_idx
                ]
                modality_idx += 1
            unit_idx += 1

        self._decoders[name].add_weights(
            weights_by_bin, weights_by_modality, weights_by_cell
        )

        return  # TODO: This function may not be complete

    def calculate_correlations(self):
        pass

    def save(self, path=""):
        pass

    def _make_bins(self, bin_stop, bin_start, bin_width, stim_table):
        if bin_stop == 0:
            bin_stop = stim_table.duration.mean() + bin_width
        return np.arrange(bin_start, bin_stop, bin_width)

    def _sort_by_modality(self, stim_modalities, stim_presentations, stim_ids):
        indicies = {}
        num_presentations = len(stim_presentations)
        if not len(stim_ids) == num_presentations:
            raise ValueError(
                f"The number of stimulus presentations did not match the number of stimulus presentation IDs."
            )

        # For every way the stim could be presented
        for stim in stim_modalities:
            currentIDs = []
            # Collect all the stimulus IDs that match the stim
            for presentation in range(num_presentations):
                if stim_presentations[presentation] == float(stim):
                    currentIDs = currentIDs + [stim_ids[presentation]]
                # FIXME: There may be a bug here with the datatype of the stimulus
                # If it becomes an issue, pull the presentation type and cast the
                # stim to that type
            indicies[stim] = currentIDs

        return indicies

    def _initialize_dict(self, keys, array_size):
        empty_dict = {}
        for key in keys:
            empty_dict[key] = np.zeros(array_size)
        return empty_dict
