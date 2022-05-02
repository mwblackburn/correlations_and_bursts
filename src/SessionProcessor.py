import numpy as np
import pandas as pd
from numpy.random import default_rng
import xarray
import copy

import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.svm import LinearSVC
from sklearn import metrics
from scipy.stats import pearsonr as pearson_correlation

from src.Decoder import Decoder

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

# For visualization: scripts for plotting
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

    def __init__(self, session, bursts=None):
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
        self._results = None

        return

    # In theory, you could cause a bug here by calling this twice, with the same stim and bin
    # width, but different start and stop times. Right now, this issue is addressed with an
    # exception (ValueError)
    def construct_decoder(
        self,
        stimulus_type,
        stimulus_characteristic,
        name="",
        bin_start=0.0,
        bin_stop=0.0,
        bin_width=0.05,
        classifier=LinearSVC(),
        burst_dict=None,
        single_dict=None,
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
            name = f"{stimulus_type}__width_{bin_width*1000}ms__{name_extension}"

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
        y = np.array(stim_table[stimulus_characteristic])

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

        if (burst_dict is not None) and (single_dict is not None):
            self.add_bursts(burst_dict, single_dict, name, shuffle_trials)

        return name

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
        return

    def calculate_decoder_weights(
        self, name, test_size=0.2, thorough_accuracy_scoring=False, cv_count=5
    ):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        # TODO: Add burst, non burst, and 2 channel functionality
        # General plan: Check that bursts and singles are present
        # If they are: do everything identically
        # Solution: make this function a wrapper function that
        # passes the x and y data directly, so that it can be
        # called with x=bursts, then called again with x=singles,
        # and so on

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

        if thorough_accuracy_scoring:
            thorough_accuracy_scores = {}
        else:
            thorough_accuracy_scores = None

        # Get data information
        num_presentations, num_bins, num_units = x.shape
        y_true = y.astype(int)

        # Initialize everything
        weights_by_modality = self._initialize_dict(
            stim_modalities, (num_bins, num_units)
        )
        weights_by_cell = self._initialize_dict(
            self.all_units, (num_bins, len(stim_modalities))
        )
        weights_by_bin = {}
        accuracies_by_bin = {}

        # Train the classifier by bin, then store the resulting weights
        for bin in range(num_bins):
            # Get the data for the current time bin
            x_bin = x[:, bin, :]

            # Split the data
            x_train, x_test, y_train, y_test = train_test_split(
                x_bin, y_true, test_size=test_size
            )

            # Train the classifier
            classifier.fit(x_train, y_train)

            # Store the weights, and the classes.
            # The classes must be stored so that the correct set of
            # weights can be associated with the correct stim
            bin_weights = classifier.coef_
            classes = classifier.classes_
            weights_by_bin[bin] = bin_weights

            accuracies_by_bin[bin] = metrics.precision_score(
                y_test, classifier.predict(x_test), average="micro"
            )  # classifier.score(x_bin, y_true)

            if thorough_accuracy_scoring:
                thorough_accuracy_scores[bin] = cross_val_score(
                    classifier,
                    x_bin,
                    y_true,
                    cv=cv_count,
                    scoring=make_scorer(accuracy_score),
                )

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
            weights_by_bin,
            weights_by_modality,
            weights_by_cell,
            accuracies_by_bin,
            thorough_accuracy_scores,
        )

        if thorough_accuracy_scoring:
            return thorough_accuracy_scores
        else:
            return accuracies_by_bin

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
        warnings.filterwarnings("ignore")
        cell_correlation_matrices = (
            {}
        )  # The weight/activity correlation matrices by class
        within_class_correlations = (
            {}
        )  # The mean of the diagonals of the above matrices
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
        warnings.filterwarnings("default")
        return

    def add_bursts(self, burst_dict, single_dict, name, shuffled=False):

        if not name in self._decoders.keys():
            raise ValueError(
                f"{name} did not match the name of any decoders constructed by this object."
            )

        # burst_dict is a dictionary with unit ids as keys, and the whole burst
        # train for the particular stimulus epoch.
        if shuffled:
            # FIXME: This needs to be done for every unit id
            # Additionally, the presentationwise functions very likely need to be adjusted
            # so that their arguments are the exact same as the AllenSDK equivalents.
            # That will make it so that the _shuffle_trials function only needs minor adjustments
            # to be used to shuffle bursts and singles.
            # To clarify, the below shuffling code isn't even close to correct right now. In fact
            # I would expect it to crash before it even produced output
            rng = default_rng()
            stim_presentation_ids = list(burst_dict["stimulus_presentation_ids"])
            burst_dict["stimulus_presentation_ids"] = rng.shuffle(stim_presentation_ids)

            stim_presentation_ids = list(single_dict["stimulus_presentation_ids"])
            single_dict["stimulus_presentation_ids"] = rng.shuffle(
                stim_presentation_ids
            )

        self._decoders[name].add_bursts(burst_dict, single_dict)

        return

    def presentationwise_burst_count(self, name, bin_edges, stimulus_presentation_ids, unit_ids):
        # TODO: Clarify in the documentation for this function that a burst might straddle bin edges,
        # so bursts are considered to be inside the bin they start in, regardless of where they end
        if not name in self._decoders.keys():
            raise ValueError(
                f"{name} did not match the name of any decoders constructed by this object."
            )
        elif not self._decoders[name].has_bursts():
            raise ValueError(f"{name} does not have any bursts to count.")
        else:
            # (
            #     classifier,
            #     stimulus_type,
            #     stim_table,
            #     stim_modalities,
            #     bins,
            #     x,
            #     y,
            # ) = self._decoders[name].unpack()
            #stim_presentation_ids = stim_table.index.values
            #stim_presentation_id_offset = int(stim_presentation_ids[0])
            # stim_presentation_indices = stim_presentation_ids - stim_presentation_ids[0]
            burst_dict = self._decoders[name].whole_burst_dict
            #num_presentations, num_bins, num_units = x.shape
            bins = bin_edges
            stim_presentation_ids = stimulus_presentation_ids
            stim_presentation_id_offset = int(stim_presentation_ids[0])
            num_bins = len(bins)
            num_presentations = len(stim_presentation_ids)
            num_units = len(unit_ids)
            
        # Initialize the return array
        presentationwise_counts = np.zeros((num_presentations, num_bins, num_units))
        
        # Walk through each bin, and check if any of the beg times are in it. If they are,
        # presentationwise_counts[presentation_id, bin_num, unit_id] += 1
        unit_idx = 0
        #for unit_id, bursts in burst_dict.items():
        for unit_id in unit_ids:
            bursts = burst_dict[unit_id]
            if bursts is not None:
                # Initialize the "next bin" to be the first bin
                next_bin = bins[0]
                for bin_idx in range(1, num_bins):
                    # Set the current bin, and next bin
                    current_bin = next_bin
                    next_bin = bins[bin_idx]

                    # Get all the bursts that lie between the current and next bin edge
                    binned_bursts = bursts.loc[((bursts["relative_beg_time"] < next_bin)*(bursts["relative_beg_time"] >= current_bin))]

                    # Count how many bursts were in this bin
                    for presentation_id in binned_bursts["stimulus_presentation_id"]:
                        # The presentation ID may not index from 0, so we force it to by
                        # subtracting the smallest presentation_id
                        presentation_idx = int(
                            presentation_id - stim_presentation_id_offset
                        )

                        # Count how many bursts occured for this presentation (presentation_idx),
                        # in this bin (bin_idx - 1), with this unit (unit_idx)
                        presentationwise_counts[
                            presentation_idx, bin_idx - 1, unit_idx
                        ] += 1

            unit_idx += 1

        # Should return: xarray with name 'burst_counts,' and axis labels 'stimulus_presentation_id',
        # 'time_relative_to_stimulus_onset', and 'unit_id'
        return xarray.DataArray(
            data=presentationwise_counts,
            dims=[
                "stimulus_presentation_id",
                "time_relative_to_stimulus_onset",
                "unit_id",
            ],
            name="burst_counts",
        )

    def presentationwise_burst_times(self, name, stimulus_presentation_ids, unit_ids):
        if not name in self._decoders.keys():
            raise ValueError(
                f"{name} did not match the name of any decoders constructed by this object."
            )
        elif not self._decoders[name].has_bursts():
            raise ValueError(f"{name} does not have any bursts.")
        else:
            burst_dict = self._decoders[name].whole_burst_dict
            stim_id_mask = {"stimulus_presentation_id": stimulus_presentation_ids}

        # Initialize the return dataframe with appropriate column labels
        presentationwise_times = pd.DataFrame(
            {
                "absolute_beg_time": [],
                "absolute_end_time": [],
                "relative_beg_time": [],
                "relative_end_time": [],
                "stimulus_presentation_id": [],
                "unit_id": [],
            }
        )
        #for unit_id, bursts in burst_dict.items():
        for unit_id in unit_ids:
            bursts = burst_dict[unit_id]
            if bursts is not None:
                # Only check the specified stimuli (specified by stimulus_presentation_ids)
                burst_locations = bursts.isin(stim_id_mask)["stimulus_presentation_id"]
                bursts = bursts.loc[burst_locations]
                
                # Get the columns from `bursts` that describe the presentationwise burst times
                abs_beg_times = bursts["absolute_beg_time"]
                abs_end_times = bursts["absolute_end_time"]
                rel_beg_times = bursts["relative_beg_time"]
                rel_end_times = bursts["relative_end_time"]
                #stim_presentation_ids = bursts["stimulus_presentation_id"]
                # unit id needs to be included. `bursts` is organized by unit,
                # so we just need to add a column full of this particular unit id
                unit_id_arr = np.full(len(bursts), unit_id)

                # Collect everything into a dataframe
                current_unit_presenationwise_times = pd.DataFrame(
                    {
                        "absolute_beg_time": abs_beg_times,
                        "absolute_end_time": abs_end_times,
                        "relative_beg_time": rel_beg_times,
                        "relative_end_time": rel_end_times,
                        "stimulus_presentation_id": bursts["stimulus_presentation_id"],
                        "unit_id": unit_id_arr,
                    }
                )

                # Join the new dataframe with the previous one(s) created
                presentationwise_times = pd.concat(
                    [presentationwise_times, current_unit_presenationwise_times],
                    ignore_index=True,
                )

        # Present the burst times in the absolute order they occured
        presentationwise_times = presentationwise_times.sort_values(
            by=["absolute_beg_time"], ascending=True, ignore_index=True
        )
        # presentationwise_times.drop(["index"], axis=1)
        return presentationwise_times

    def presentationwise_non_burst_count(self, name, bin_edges, stimulus_presentation_ids, unit_ids):
        if not name in self._decoders.keys():
            raise ValueError(
                f"{name} did not match the name of any decoders constructed by this object."
            )
        elif not self._decoders[name].has_singles():
            raise ValueError(
                f"{name} does not have any isolated single spikes to count."
            )
        else:
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
            # stim_presentation_id_offset = int(stim_presentation_ids[0])
            # stim_presentation_indices = stim_presentation_ids - stim_presentation_ids[0]
            single_dict = self._decoders[name].whole_single_dict
            #num_presentations, num_bins, num_units = x.shape
            bins = bin_edges
            stim_presentation_id_offset = int(stimulus_presentation_ids[0])
            num_bins = len(bins)
            num_presentations = len(stimulus_presentation_ids)
            num_units = len(unit_ids)

        # Initialize the return array
        presentationwise_counts = np.zeros((num_presentations, num_bins, num_units))

        # Walk through each bin, and check if any of the spike times are in it. If they are,
        # presentationwise_counts[presentation_id, bin_num, unit_id] += 1
        unit_idx = 0
        #for unit_id, singles in single_dict.items():
        for unit_id in unit_ids:
            singles = single_dict[unit_id]
            if singles is not None:
                # Initialize the "next bin" to be the first bin
                next_bin = bins[0]
                for bin_idx in range(1, num_bins):
                    # Set the current bin, and next bin
                    current_bin = next_bin
                    next_bin = bins[bin_idx]

                    # Get all the singles that lie between the current and next bin edge
                    binned_singles = singles.loc[
                        ((singles["relative_spike_time"] < next_bin)*(singles["relative_spike_time"] >= current_bin))
                    ]

                    # Count how many bursts were in this bin
                    for presentation_id in binned_singles["stimulus_presentation_id"]:
                        # The presentation ID may not index from 0, so we force it to by
                        # subtracting the smallest presentation_id
                        presentation_idx = int(
                            presentation_id - stim_presentation_id_offset
                        )

                        # Count how many bursts occured for this presentation (presentation_idx),
                        # in this bin (bin_idx - 1), with this unit (unit_idx)
                        presentationwise_counts[
                            presentation_idx, bin_idx - 1, unit_idx
                        ] += 1

            unit_idx += 1

        # Should return: xarray with name 'single_spike_counts,' and axis labels 'stimulus_presentation_id',
        # 'time_relative_to_stimulus_onset', and 'unit_id'
        return xarray.DataArray(
            data=presentationwise_counts,
            dims=[
                "stimulus_presentation_id",
                "time_relative_to_stimulus_onset",
                "unit_id",
            ],
            name="single_spike_counts",
        )

    def presentationwise_non_burst_times(self, name, stimulus_presentation_ids, unit_ids):
        if not name in self._decoders.keys():
            raise ValueError(
                f"{name} did not match the name of any decoders constructed by this object."
            )
        elif not self._decoders[name].has_singles():
            raise ValueError(f"{name} does not have any isolated single spikes.")
        else:
            single_dict = self._decoders[name].whole_single_dict
            stim_id_mask = {"stimulus_presentation_id": stimulus_presentation_ids}
        
        # Initialize the return dataframe with appropriate column labels
        presentationwise_times = pd.DataFrame(
            {
                "absolute_spike_time": [],
                "relative_spike_time": [],
                "stimulus_presentation_id": [],
                "unit_id": [],
            }
        )
        #for unit_id, singles in single_dict.items():
        for unit_id in unit_ids:
            singles = single_dict[unit_id]
            if singles is not None:
                # Only check the specified stimuli (specified by stimulus_presentation_ids)
                single_locations = singles.isin(stim_id_mask)["stimulus_presentation_id"]
                singles = singles.loc[single_locations]
                
                # Get the columns from `singles` that describe the presentationwise burst times
                abs_spike_times = singles["absolute_spike_time"]
                rel_spike_times = singles["relative_spike_time"]
                stimulus_presentation_ids = singles["stimulus_presentation_id"]
                # unit id needs to be included. `singles` is organized by unit,
                # so we just need to add a column full of this particular unit id
                unit_ids = np.full(len(singles), unit_id)

                # Collect everything into a dataframe
                current_unit_presenationwise_times = pd.DataFrame(
                    {
                        "absolute_spike_time": abs_spike_times,
                        "relative_spike_time": rel_spike_times,
                        "stimulus_presentation_id": singles["stimulus_presentation_id"],
                        "unit_id": unit_ids,
                    }
                )

                # Join the new dataframe with the previous one(s) created
                presentationwise_times = pd.concat(
                    [presentationwise_times, current_unit_presenationwise_times],
                    ignore_index=True,
                )

        # Present the burst times in the absolute order they occured
        presentationwise_times = presentationwise_times.sort_values(
            by=["absolute_spike_time"], ascending=True, ignore_index=True
        )
        # presentationwise_times.drop(["index"], axis=1)
        return presentationwise_times

    # TODO: Implement the below functions
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

    def results(self):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        # TODO: There will be other things that need checking (make sure proper calls have been made, etc.)
        # if self._results is not None:
        #    return self._results

        # self._decoders = {}
        # self._histograms = {}
        # self._modality_histograms = {}
        # self._cell_correlations = {}
        # self._within_class_correlations = {}
        names = self._decoders.keys()
        results = {}
        for name in names:
            name_results = {}
            name_results["decoder"] = self._decoders[name]
            # name_results["decoder_accuracy_by_bin"] = ?
            name_results["psths"] = self._histograms[name]
            name_results["class_psths"] = self._modality_histograms[name]
            name_results["cell_correlation_matrices"] = self._cell_correlations[name]
            name_results["within_class_correlations"] = self._within_class_correlations[
                name
            ]
            results[name] = name_results

        self._results = results
        return results

    # (bins, stim_ids, y, stim_modalities)
    # TODO: This function calls "presentationwise_spike_counts" to shuffle the cell activity.
    # The easiest way to adjust this function s.t. it will shuffle singles and bursts is to
    # adjust the "presentationwise" functions to take the exact same arguments (and the 
    # `name`) so that the local "presentationwise" functions can be called the same way 
    # as the AllenSDK functions are
    def _shuffle_trials(
        self, bin_edges, stim_ids, stim_presentation_order, stim_classes
    ):
        """Brief summary of what this function does.
        
        Note
        ____
        There's probably something to note here
        
        Parameters
        __________
        
        
        Returns
        _______
        
        """
        # Overall method:
        # Isolate PSTHs by class (so collect all the responses to 45 degrees in one spot,
        # 90 in another, etc). With the PSTHs isolated by class, shuffling them is easier
        # because we then just have to loop through bins and units to shuffle.

        rng = default_rng()
        num_bins = len(bin_edges) - 1
        num_presentations = len(stim_presentation_order)
        num_units = len(self.all_units)

        # Sort all the stimulus presentation ids by class so that trials are shuffled with
        # the correct labels
        presentations_by_class = {}
        counts = {}
        for (
            stim_class
        ) in stim_classes:  # For each class of stimulus (e.g. each presentation angle)
            class_presentation_indicies = []
            counts[
                stim_class
            ] = 0  # Used later when collecting shuffled presentations in order
            # Collect the indicies for each presentation of that class
            for k in range(num_presentations):
                if stim_class == stim_presentation_order[k]:
                    class_presentation_indicies = class_presentation_indicies + [k]
            # presentations_by_class[stim_class] = class_presentation_indicies
            # The psths for every presentation for every cell, sorted by stimulus
            presentations_by_class[stim_class] = np.array(
                self.session.presentationwise_spike_counts(
                    bin_edges, stim_ids[class_presentation_indicies], self.all_units
                )
            )

        # PSTHS: (num_presentations, num_bins, num_units)
        # We want to loop through each bin -> current_bin: (num_presentations, num_units)
        # Shuffle each column of current_bin (keeps units' responses within unit, but outside of trial)
        for stim_class in stim_classes:
            current_class_psth = presentations_by_class[stim_class]
            for bin_idx in range(num_bins):
                rng.shuffle(current_class_psth[:, bin_idx, :], axis=0)
                # current_bin = current_class_psth[:,bin_idx,:]
                # Shuffle the current bin column wise
                # current_class_psth[:,bin_idx,:] = rng.shuffle(current_bin, axis=1)

        # Now all the stimulus presentations are shuffled within class, and we need to collect them
        # all into one array (with the original class presentation ordering given by
        # stim_presentation_order)
        presentation_idx = 0
        psths = np.zeros((num_presentations, num_bins, num_units))
        for presentation_class in stim_presentation_order:
            # Get the psth stack for this class
            current_class_psth = presentations_by_class[presentation_class]
            # Get how many times we've seen this class before
            current_class_count = counts[presentation_class]
            # Add the next presentation to the all class psth stack
            psths[presentation_idx] = current_class_psth[current_class_count]

            # Increment indices
            counts[presentation_class] += 1
            presentation_idx += 1

        # Create the data to be shuffled
        # psths = self.session.presentationwise_spike_counts(
        #     bin_edges, stim_ids, self.all_units
        # )
        # num_presentations, num_bins, num_units = psths.shape

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
        # for presentation_idx in range(num_presentations):
        #     # current_presentation = psths[presentation_idx, :, :]
        #     current_class = stim_presentation_order[presentation_idx]

        #     # This is the substack I refer to above
        #     current_class_indices = presentations_by_class[current_class]
        #     num_class_presentations = len(current_class_indices)

        #     # We're going to need to swap (num_bins*num_units) entries.
        #     # This is a matrix of random indicies for the substack
        #     swapping_partner_indices = np.random.random_integers(
        #         0, num_class_presentations - 1, (num_bins, num_units)
        #     )

        #     for bin_idx in range(num_bins):
        #         for unit_idx in range(num_units):
        #             # swapping_partner_indices[bin_idx, unit_idx] -> the index of the random
        #             # [m,n]th entry in the substack. Name it IDX
        #             # current_class_indices[IDX] -> the index of the random [m,n]th entry in
        #             # the entire whole stack
        #             # -> swapping_partner_idx is a random index of the same class as the
        #             # current class
        #             swapping_partner_idx = current_class_indices[
        #                 swapping_partner_indices[bin_idx, unit_idx]
        #             ]

        #             # Hold the value at the current slice
        #             switch_bag = psths[presentation_idx, bin_idx, unit_idx]

        #             # Replace the value at the current slice with the value at the random index
        #             psths[presentation_idx, bin_idx, unit_idx] = psths[
        #                 swapping_partner_idx, bin_idx, unit_idx
        #             ]

        #             # Replace the value at the random index with the value at the current slice
        #             psths[swapping_partner_idx, bin_idx, unit_idx] = switch_bag

        return psths

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
        warnings.filterwarnings("ignore")

        correlations = np.zeros(x1.shape[0])
        for k in range(x1.shape[0]):
            correlations[k] = pearson_correlation(x1[k, :], x2)[0]

        warnings.filterwarnings("default")
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
        keys = list(histograms.keys())
        num_bins = histograms[keys[0]].shape[0]

        # Convert the dictionary of histograms to an array:
        # We're pulling a single cell's histograms, so the shape of the array should be (num_bins, num_modalities)
        return_histograms = np.zeros((num_bins, len(keys)))

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
                current_correlation = pearson_correlation(currentWeights, current_psth)[
                    0
                ]
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
