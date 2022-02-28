import numpy as np

# import warnings
from sklearn.svm import LinearSVC
from src.Decoder import Decoder
from scipy.stats import pearsonr as pearson_correlation


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
            start = len(
                self.all_units
            )  # Because self.all_units is constantly growing, this will change every iteration
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
        self._cell_correlations = {}

        # TODO: Construct more variables as needed
        return

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

        # Generate the histogram
        self._histograms[name] = self.session.presentationwise_spike_counts(
            bins, stim_table.index.values, self.all_units
        )

        # Sort the stimulus presentations by modality
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

    def calculate_correlations(self, name):
        if not name in self._decoders.keys():
            raise ValueError(
                f"{name} did not match the name of any decoders constructed by this object."
            )
        elif not name in self._histograms.keys():
            raise ValueError(
                f"{name} did not match the name of any histograms constructed by this object."
            )
        elif not self._decoders[name].has_weights():
            raise ValueError(
                f"{name} has both a decoder and PSTHs, but the decoder has no weights."
            )
        else:
            # Most of the below code may be unneeded, remove it if its still unneeded after testing
            # (
            #     classifier,
            #     stimulus_type,
            #     stim_table,
            #     stim_modalities,
            #     bins,
            #     x,
            #     y,
            # ) = self._decoders[name].unpack()
            # stim_presentation_ids = stim_table.index.values
            weights_by_bin, weights_by_modality, weights_by_cell = self._decoders[
                name
            ].unpack_weights()
            # histograms = self._histograms[name]
            # M number of stimulus modalities -> need the mean histogram across the stim modalities
            modality_histograms = {}  # self._modality_histograms[name]
            for stim in self._modality_histograms[name].keys():
                modality_histograms[stim] = np.array(
                    self._modality_histograms[name][stim].mean(
                        dim="stimulus_presentation_id"
                    )
                )

        cell_correlation_matrices = {}
        cell_idx = 0
        for cell_id in self.all_units:
            cell_weights = weights_by_cell[cell_id]
            # Get the psth for the current cell for each modality (arr: (num_bins, num_modalities))
            cell_histograms = self._organize_histograms(modality_histograms, cell_idx)
            cell_correlation_matrices[cell_id] = self._correlate_by_cell(
                cell_weights, cell_histograms
            )
            cell_idx += 1
        # TODO: Calculate the diagonals and save them
        self._cell_correlations[name] = cell_correlation_matrices
        return

    def trial_shuffle(self):
        # Currently this function is here as a placeholder for potentially multiple functions
        # related to trial shuffling
        pass

    def save(self, path=""):
        pass

    def _correlate(self, x1, x2):
        # warnings.filterwarnings('ignore')

        correlations = np.zeros(x1.shape[0])
        for k in range(x1.shape[0]):
            correlations[k] = pearson_correlation(x1[k, :], x2)[0]

        # warnings.filterwarnings('default')
        return np.nanmean(correlations)

    def _organize_histograms(self, histograms, cell_idx):
        keys = histograms.keys()
        num_bins = histograms[keys[0]].shape[0]

        # Convert the dictionary of histograms to an array:
        # We're pulling a single cell's histograms, so the shape of the array should be (num_bins, num_modalities)
        return_histograms = np.zeros((num_bins, len(keys)))

        # FIXME: This loop probably has a bug in it, most likely something to do with indexing
        k = 0
        for key in keys:
            return_histograms[:, k] = histograms[key][:, cell_idx]
            k += 1
        return return_histograms

    # modality_histograms here is a 2D array, where rows are bins, and columns are modalities for a specific cell
    def _correlate_by_cell(self, cell_weights, modality_histograms):
        num_modalities = modality_histograms.shape[1]
        cell_correlations = np.zeros((num_modalities, num_modalities))

        # One loop goes over the rows of the cell_correlations matrix, the other over the columns.
        # The loop should fill the matrix with correlation coefficients between the average PSTH for a
        # specific modality and the decoder weights associated with that modality
        # Ideally, we want large diagonals and small off-diagonals (high correlation between weights of
        # a specific modality and the activity of that modality)
        for row in range(num_modalities):
            currentWeights = cell_weights[:, row]
            for col in range(num_modalities):
                current_psth = modality_histograms[:, col]
                # If warnings become an issue, change the pearson_correlation call to a self._correlate call
                current_correlation = pearson_correlation(currentWeights, current_psth)
                cell_correlations[row, col] = (
                    current_correlation if current_correlation is not np.nan else 0
                )

        return cell_correlations

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

    # Initializes a dictionary full of empty arrays
    def _initialize_dict(self, keys, array_size):
        empty_dict = {}
        for key in keys:
            empty_dict[key] = np.zeros(array_size)
        return empty_dict
