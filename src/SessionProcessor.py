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

# BRING UP: Refactor modality -> class (ie weights_by_modality -> weights_by_class)
# BRING UP: Include functionality for decoder optimization -> Yes, port the optimization stuff over
# BRING UP: Docstring format for dictionaries

# For visualization: Either another class, or scripts for plotting
# Script for running analysis and using the objects (part of both documentation and testing)
# Make script for replicating shown analyses (also part of both documentation and testing)


class SessionProcessor:
    # Currently irrelevant documentation that was dificult to phrase, so I'm waiting to delete it
    # until I'm certain I wont need it
    #        The indices for the beginning of each region's unit ID numbers is held in
    #    _all_unit_dividing_indicies
    """One line summary of SessionProcessor
    
    More in depth description of SessionProcessor
    
    Methods
    _______
    construct_decoder(LIST_ALL_ARGS)
        Description of construct_decoder
    construct_psth(LIST_ALL_ARGS)
        Description of construct_psth
    etc.
    
    Attributes
    __________
    session : EcephysSession
        The EcephysSession this processor navigates.
    units_by_acronym : :obj:`dict` of :obj:`list` of :obj:`int`
        Holds the unit ID numbers of every cell belonging to a specific region.
        The region (acronym) is the key for the associated list of ID numbers.
    all_units : :obj:`list` of :obj:`int`
        The unit ID number of every cell in this session. all_units is sorted by
        region, e.g. all 'VISp' units lie between all_units[i] and all_units[j].
    
    """

    # TODO: Doc me

    def __init__(self, session):
        """Create a new SessionProcessor
        
        Note
        ____
        Maybe theres something to note?
        
        Parameters
        __________
        session : EcephysSession
            The EcephysSession to be processed.
        
        """

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
        # the decoders, PSTHs, and correlations
        self._decoders = {}
        self._histograms = {}
        self._modality_histograms = {}
        self._cell_correlations = {}
        self._within_class_correlations = {}

        # TODO: Construct more variables as needed
        return

    # In theory, you could cause a bug here by calling this twice, with the same stim and bin
    # bin width, but different start and stop times. Right now, this issue is addressed with
    # an exception (ValueError)
    def construct_decoder(
        self,
        stimulus_type,
        name="",
        bin_start=0.0,
        bin_stop=0.0,
        bin_width=0.05,
        classifier=LinearSVC(),
        shuffle_trials=False,
    ) -> None:
        """Collects all the data needed to decode stimuli from the neural activity in self.session.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        # If a name wasn't provided, name the decoder as explicitly as possible
        if name == "":
            name_extension = "shuffled" if shuffle_trials else "unshuffled"
            name = f"{stimulus_type}_width_{bin_width*1000}ms_{name_extension}"

        if name in self._decoders.keys():
            raise ValueError(
                f"A decoder with the name {name}, already exists. This can happen if you have constructed a decoder with the same stimulus_type, bin_width, and shuffle_trials."
            )

        # Get the stim info
        stim_table = self.session.get_stimulus_table(stimulus_type)
        stim_ids = stim_table.index.values

        # Create the time bins
        bins = self._make_bins(bin_start, bin_stop, bin_width, stim_table)

        # The labels to be predicted by the decoder
        y = np.array(stim_table[stimulus_type])

        # Change the null labels to a numerical value, so that the classifiers don't panic
        for idx in range(len(y)):
            if y[idx] == "null":
                y[idx] = -1.0
        stim_modalities = np.unique(y)

        # Shuffle x if need be
        if shuffle_trials:
            # Shuffle trials within classes
            x = self._shuffle_trials(bins, stim_ids, y, stim_modalities)
        else:
            # Collect the data
            x = self.session.presentationwise_spike_counts(
                bins, stim_ids, self.all_units
            )
        x = np.array(x)

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
        return  # NOTE: This function may not be complete

    # Idea: Eventually what we want, is to pass a set of stim names, and return psths, weights,
    # correlations, etc for every stim

    # Must be called after construct_decoder is called.
    # XXX: Maybe adjust it so that it gives a warning but constructs a default decoder
    # with the name that was passed as an argument?
    # BRING UP: There is a lot of redundant code here that could easily be combined with
    # the code in "construct decoder." The easiest thing to do would be to modify this
    # function to be private, and pass the relevant arguments to this function and
    # construct the psths. You could then pretty easily associate a set of psths with
    # a specific decoder
    def construct_psth(self, name):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
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

    # BRING UP: This also might be overly redundant, why not just call it immediately after
    # construct_decoder(args) is called?
    def calculate_decoder_weights(self, name):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
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

        return  # NOTE: This function may not be complete

    def calculate_correlations(self, name):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
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
        within_class_correlations = {}
        cell_idx = 0
        for cell_id in self.all_units:
            cell_weights = weights_by_cell[cell_id]
            # Get the psth for the current cell for each modality (arr: (num_bins, num_modalities))
            cell_histograms = self._organize_histograms(modality_histograms, cell_idx)
            (
                cell_correlation_matrices[cell_id],
                within_class_correlations[cell_id],
            ) = self._correlate_by_cell(cell_weights, cell_histograms)
            cell_idx += 1
        self._cell_correlations[name] = cell_correlation_matrices
        self._within_class_correlations[name] = within_class_correlations
        # TODO: Histogram the diagonals in bulk and by region
        return

    # (bins, stim_ids, y, stim_modalities)
    def _shuffle_trials(self, bin_edges, stim_ids, stim_presentations, stim_classes):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        # Create the data to be shuffled
        psths = self.session.presentationwise_spike_counts(
            bin_edges, stim_ids, self.all_units
        )
        num_presentations, num_bins, num_units = psths.shape

        # Sort all the stimulus presentation ids by class so that trials are shuffled with
        # the correct labels
        presentations_by_class = {}
        for stim_class in stim_classes:
            class_presentation_indicies = []
            for k in range(num_presentations):
                if stim_class == stim_presentations[k]:
                    class_presentation_indicies = class_presentation_indicies + [k]
            presentations_by_class[stim_class] = class_presentation_indicies

        # Picture a psth set that is num_bins x num_units. That is the psth set for a particular
        # stimulus presentation. I'm going to refer to that as the "current slice."
        # There is a stack of these slices. A slice has an associated stimulus presentation in
        # stim_presentations, which allows us to group the indices of the slices by presentation
        # class, e.g. given classes A and B, all the slice indices for class A are grouped and
        # all the slice indicies for class B are grouped. I'll refer to the class grouped stacks
        # as "substacks."
        # We're taking the current slice, and looping through every entry in it.
        # At the [m,n]th entry in the slice, we pull a random [m,n]th entry from the associated
        # substack and swap them.
        for presentation_idx in range(num_presentations):
            # current_presentation = psths[presentation_idx, :, :]
            current_class = stim_presentations[presentation_idx]

            # This is the substack I refer to above
            current_class_indices = presentations_by_class[current_class]
            num_class_presentations = len(current_class_indices)

            # We're going to need to swap (num_bins*num_units) entries.
            # This is a matrix of random indicies for the substack
            swapping_partner_indices = np.random.random_integers(
                0, num_class_presentations - 1, (num_bins, num_units)
            )

            for bin_idx in range(num_bins):
                for unit_idx in range(num_units):
                    # swapping_partner_indices[bin_idx, unit_idx] -> the index of the random
                    # [m,n]th entry in the substack. Name it IDX
                    # current_class_indices[IDX] -> the index of the random [m,n]th entry in
                    # the entire whole stack
                    # -> swapping_partner_idx is a random index of the same class as the
                    # current class
                    swapping_partner_idx = current_class_indices[
                        swapping_partner_indices[bin_idx, unit_idx]
                    ]

                    # Hold the value at the current slice
                    switch_bag = psths[presentation_idx, bin_idx, unit_idx]

                    # Replace the value at the current slice with the value at the random index
                    psths[presentation_idx, bin_idx, unit_idx] = psths[
                        swapping_partner_idx, bin_idx, unit_idx
                    ]

                    # Replace the value at the random index with the value at the current slice
                    psths[swapping_partner_idx, bin_idx, unit_idx] = switch_bag

        return psths

    def save(self, path=""):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        pass

    def _correlate(self, x1, x2):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        # warnings.filterwarnings('ignore')

        correlations = np.zeros(x1.shape[0])
        for k in range(x1.shape[0]):
            correlations[k] = pearson_correlation(x1[k, :], x2)[0]

        # warnings.filterwarnings('default')
        return np.nanmean(correlations)

    def _organize_histograms(self, histograms, cell_idx):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
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
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
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

        # Pull average diagonal (within class correlations)
        diagonal = 0.0
        for k in range(num_modalities):
            diagonal = diagonal + cell_correlations[k, k]
        diagonal = diagonal / num_modalities

        return cell_correlations, diagonal

    def _make_bins(self, bin_stop, bin_start, bin_width, stim_table):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        if bin_stop == 0:
            bin_stop = stim_table.duration.mean() + bin_width
        return np.arange(bin_start, bin_stop, bin_width)

    def _sort_by_modality(self, stim_modalities, stim_presentations, stim_ids):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
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
                # NOTE: There may be a bug here with the datatype of the stimulus
                # If it becomes an issue, pull the presentation type and cast the
                # stim to that type
            indicies[stim] = currentIDs

        return indicies

    # Initializes a dictionary full of empty arrays
    def _initialize_dict(self, keys, array_size):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        empty_dict = {}
        for key in keys:
            empty_dict[key] = np.zeros(array_size)
        return empty_dict
