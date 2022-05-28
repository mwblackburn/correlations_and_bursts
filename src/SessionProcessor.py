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

from src.Decoder import DataInstance

# import allensdk.brain_observatory.ecephys.ecephys_session


# This is where most of the calculations on a single session are going to be performed.
# It will:
# construct and evaluate decoders
# construct and evaluate PSTHs
# construct and evaluate correlation statistics
# save any relevant data for plots (so basically everything)


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
            # Because self.all_units is constantly growing, this will change every iteration
            start = len(self.all_units)
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
    ) -> str:
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
            x = self._shuffle_trials(
                bins,
                stim_ids,
                y,
                stim_modalities,
                name,
                self._presentationwise_spike_counts,
            )
        else:
            # Collect the data
            x = self.session.presentationwise_spike_counts(
                bins, stim_ids, self.all_units
            )
        x = np.array(x)

        # Construct the decoder
        self._decoders[name] = DataInstance(
            classifier,
            stimulus_type,
            stim_table,
            stim_modalities,
            bins,
            x,
            y,
            shuffle_trials,
            name=name,
        )

        if (burst_dict is not None) and (single_dict is not None):
            self.add_bursts(burst_dict, single_dict, name, shuffle_trials)

        return name

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

        # Sort the stimulus presentations by modality
        modality_indicies = self._sort_by_modality(
            stim_modalities, y, stim_presentation_ids
        )

        # Generate the histograms
        histograms = {}

        # The allensdk version of the presentationwise function cannot take shuffling
        # into account, so we need to manually shuffle the whole psth if the data is
        # shuffled. The processor's presentationwise functions take shuffling into
        # account, so the same check doesn't need to be made for the burst and
        # single spike psths
        histograms["whole"] = (
            self._shuffle_trials(
                bins,
                stim_presentation_ids,
                y,
                stim_modalities,
                name,
                self._presentationwise_spike_counts,
            )
            if self._decoders[name].is_shuffled()
            else self.session.presentationwise_spike_counts(
                bins, stim_presentation_ids, self.all_units
            )
        )
        histograms["bursts"] = (
            self.presentationwise_burst_counts(
                name, bins, stim_presentation_ids, self.all_units
            )
            if self._decoders[name].has_bursts()
            else None
        )
        histograms["singles"] = (
            self.presentationwise_non_burst_counts(
                name, bins, stim_presentation_ids, self.all_units
            )
            if self._decoders[name].has_singles()
            else None
        )
        self._histograms[name] = histograms

        # Generate stimulus condition (modality) specific histograms
        whole_modality_histograms = {}
        burst_modality_histograms = {} if self._decoders[name].has_bursts() else None
        single_modality_histograms = {} if self._decoders[name].has_singles() else None

        # For each possible stimulus condition
        for stim in stim_modalities:
            # self._check_membership(members, arr) returns a boolean array
            # new_arr of length len(arr) -> new_arr[k] = arr[k] in members

            # Generate the histograms
            whole_modality_histograms[stim] = histograms["whole"].loc[
                self._check_membership(
                    modality_indicies[stim],
                    histograms["whole"].stimulus_presentation_id,
                )
            ]
            burst_modality_histograms[stim] = (
                histograms["bursts"].loc[
                    self._check_membership(
                        modality_indicies[stim],
                        histograms["bursts"].stimulus_presentation_id,
                    )
                ]
                if self._decoders[name].has_bursts()
                else None
            )
            single_modality_histograms[stim] = (
                histograms["singles"].loc[
                    self._check_membership(
                        modality_indicies[stim],
                        histograms["singles"].stimulus_presentation_id,
                    )
                ]
                if self._decoders[name].has_singles()
                else None
            )

        modality_histograms = {}
        modality_histograms["whole"] = whole_modality_histograms
        modality_histograms["bursts"] = burst_modality_histograms
        modality_histograms["singles"] = single_modality_histograms
        self._modality_histograms[name] = modality_histograms
        return (self._histograms, self._modality_histograms)

    def calculate_decoder_weights(
        self, name, test_size=0.2, cv_accuracy_scoring=False, cv_count=5
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

        # Initialize variable for cross validation scoring
        if cv_accuracy_scoring:
            cv_accuracy_scores = {}
        else:
            cv_accuracy_scores = None

        # Collect the decoding inputs (whole spike trains, along with bursts and singles if available)
        data_dict = {}
        data_dict["whole"] = x
        data_labels = y.astype(int)
        if self._decoders[name].has_bursts():
            data_dict["bursts"] = self._decoders[name].burst_counts
            data_dict["singles"] = self._decoders[name].single_counts
            # data_dict["two_channel"] -> likely a different function entirely, because the random sampling needs to be repeated

        # Collect the weights, accuracy scores, and cv_accuracy scores if specified
        weights = {}
        accuracy_scores = {}
        for spike_train_type, spike_train in data_dict.items():
            # spike_train_type: "whole" "bursts" "singles" (and maybe "two_channel" later)
            (
                weights_by_bin,
                weights_by_modality,
                weights_by_cell,
                accuracies,
                thorough_accuracy_scores,
            ) = self._calculate_decoder_weights(
                spike_train,
                data_labels,
                stim_modalities,
                classifier,
                test_size,
                cv_count,
            )
            weights[spike_train_type] = weights_by_cell
            accuracy_scores[spike_train_type] = accuracies
            if cv_accuracy_scoring:
                cv_accuracy_scores[spike_train_type] = thorough_accuracy_scores

        self._decoders[name].add_weights(weights, accuracy_scores, cv_accuracy_scores)

        return

    def _calculate_decoder_weights(
        self, data, labels, stim_modalities, classifier, test_size, cv_count,
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
        if cv_count > 1:
            thorough_accuracy_scores = {}
        else:
            thorough_accuracy_scores = None
        # Get data information
        num_presentations, num_bins, num_units = data.shape
        # labels = labels.astype(int)
        # `labels` used to be `y_true`

        # Initialize everything
        weights_by_modality = self._initialize_dict(
            stim_modalities, (num_bins, num_units)
        )
        weights_by_cell = self._initialize_dict(
            self.all_units, (num_bins, len(stim_modalities))
        )
        weights_by_bin = {}
        accuracies_by_bin = {}
        cv_scorer = make_scorer(accuracy_score)

        # Train the classifier by bin, then store the resulting weights
        for bin in range(num_bins):
            # Get the data for the current time bin
            x_bin = data[:, bin, :]

            # Split the data
            x_train, x_test, y_train, y_test = train_test_split(
                x_bin, labels, test_size=test_size
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

            if thorough_accuracy_scores is not None:
                thorough_accuracy_scores[bin] = cross_val_score(
                    classifier, x_bin, labels, cv=cv_count, scoring=cv_scorer,
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
        return (
            weights_by_modality,
            weights_by_bin,
            weights_by_cell,
            accuracies_by_bin,
            thorough_accuracy_scores,
        )

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

            all_weights = self._decoders[name].unpack_weights()
            # histograms = self._histograms[name]
            # M number of stimulus modalities -> need the mean histogram across the stim modalities
            modality_histograms = {
                "whole": {},
                "bursts": {},
                "singles": {},
            }  # self._modality_histograms[name]

            # The values in modality_histograms are the PSTHs for each stimulus condition
            for spike_train_type in all_weights.keys():
                for stim in self._modality_histograms[name][spike_train_type].keys():
                    modality_histograms[spike_train_type][stim] = np.array(
                        self._modality_histograms[name][spike_train_type][stim].mean(
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

        for spike_train_type, weights in all_weights.items():
            # cstt -> current_spike_train_type
            cstt_cell_correlation_matrices = {}
            cstt_within_class_correlations = {}
            cell_idx = 0
            for cell_id in self.all_units:
                cell_weights = weights[cell_id]
                # Get the psth for the current cell for each modality (arr: (num_bins, num_modalities))
                cell_histograms = self._organize_histograms(
                    modality_histograms[spike_train_type], cell_idx
                )
                (
                    cstt_cell_correlation_matrices[cell_id],
                    cstt_within_class_correlations[cell_id],
                ) = self._correlate_by_cell(cell_weights, cell_histograms)

                cell_correlation_matrices[
                    spike_train_type
                ] = cstt_cell_correlation_matrices
                within_class_correlations[
                    spike_train_type
                ] = cstt_within_class_correlations
                cell_idx += 1

        self._cell_correlations[name] = cell_correlation_matrices
        self._within_class_correlations[name] = within_class_correlations
        warnings.filterwarnings("default")
        return

    def add_bursts(self, burst_dict, single_dict, name, shuffled=False):
        # TODO: Its probably a good idea to check `name` for the words "shuffled" or "unshuffled"
        # and shuffle the bursts based on that (keep the `shuffled` bool so that names can still
        # be user defined)
        # TODO: Clarify in the documentation that if you wan't to change the time bins for a
        # shuffled trial, you have to make a whole new decoder with those specified bins
        # (emphasize that you do not need to make another processor, call
        # construction_decoder again with the specified bins)
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
            num_presentations = x.shape[0]
            stim_presentation_ids = stim_table.index.values

        # The presentationwise functions checks for a burst or single dict in
        # self._decoders[name], and it checks whether the data has been
        # shuffled, so we add the dicts here, and say that it's unshuffled,
        # then add the results from the presentationwise function afterwards
        # and specify the shuffled status at the end
        self._decoders[name].add_bursts(burst_dict, single_dict, None, None, False)
        if shuffled:
            bursts = self._shuffle_trials(
                bins,
                stim_presentation_ids,
                y,
                stim_modalities,
                name,
                self.presentationwise_burst_counts,
            )
            bursts.name = "burst_counts"
            singles = self._shuffle_trials(
                bins,
                stim_presentation_ids,
                y,
                stim_modalities,
                name,
                self.presentationwise_non_burst_counts,
            )
            singles.name = "single_spike_counts"
        else:
            bursts = self.presentationwise_burst_counts(
                name, bins, stim_presentation_ids, self.all_units
            )
            singles = self.presentationwise_non_burst_counts(
                name, bins, stim_presentation_ids, self.all_units
            )

        # Add the burst_dict and single_dict (redundantly) then add the PSTH
        # representations of the bursts and singles
        self._decoders[name].add_bursts(
            burst_dict, single_dict, bursts, singles, shuffled
        )

        return

    def _presentationwise_spike_counts(
        self, name, bin_edges, stimulus_presentation_ids, unit_ids
    ):
        return self.session.presentationwise_spike_counts(
            bin_edges=bin_edges,
            stimulus_presentation_ids=stimulus_presentation_ids,
            unit_ids=unit_ids,
        )

    def _presentationwise_spike_times(self, name, stimulus_presentation_ids, unit_ids):
        return self.session.presentationwise_spike_times(
            stimulus_presentation_ids=stimulus_presentation_ids, unit_ids=unit_ids
        )

    def presentationwise_burst_counts(
        self, name, bin_edges, stimulus_presentation_ids, unit_ids
    ):
        # TODO: Clarify in the documentation for this function that a burst might straddle bin edges,
        # so bursts are considered to be inside the bin they start in, regardless of where they end
        if not name in self._decoders.keys():
            raise ValueError(
                f"{name} did not match the name of any decoders constructed by this object."
            )
        elif not self._decoders[name].has_bursts():
            raise ValueError(f"{name} does not have any bursts to count.")
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
            all_stim_presentation_ids = stim_table.index.values
            stim_presentation_id_offset = int(all_stim_presentation_ids[0])
            burst_dict = self._decoders[name].whole_burst_dict
            bins = bin_edges

            num_presentations, num_bins, num_units = x.shape
            if type(unit_ids) != np.ndarray:
                unit_ids = np.array(unit_ids)
            num_units = unit_ids.size

        # If this data is shuffled, this function has already been called on all the units and
        # stim presentation ids, so we can just grab the data from the decoder and return that
        # with just the specified stim_ids and unit_ids
        if self._decoders[name].is_shuffled():
            presentationwise_counts = copy.deepcopy(self._decoders[name].burst_counts)
            presentationwise_counts = presentationwise_counts.sel(
                stimulus_presentation_id=stimulus_presentation_ids, unit_id=unit_ids
            )
            return presentationwise_counts

        # Initialize the return array
        presentationwise_counts = np.zeros(
            (num_presentations, num_bins, num_units), dtype=int
        )

        # Walk through each bin, and check if any of the beg times are in it. If they are,
        # presentationwise_counts[presentation_id, bin_num, unit_id] += 1
        unit_idx = 0
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
                    binned_bursts = bursts.loc[
                        (
                            (bursts["relative_beg_time"] < next_bin)
                            * (bursts["relative_beg_time"] >= current_bin)
                        )
                    ]

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

                # Repeat the loop action for the last bin
                binned_bursts = bursts.loc[(bursts["relative_beg_time"] > next_bin)]
                for presentation_id in binned_bursts["stimulus_presentation_id"]:
                    presentation_idx = int(
                        presentation_id - stim_presentation_id_offset
                    )
                    presentationwise_counts[
                        presentation_idx, num_bins - 1, unit_idx
                    ] += 1

            unit_idx += 1

        # Take out any stim presentations not in stim_presentation_ids
        indices_to_keep = (
            np.array(stimulus_presentation_ids) - stim_presentation_id_offset
        )
        presentationwise_counts = presentationwise_counts[indices_to_keep]

        # Adjust the bin labels to be the center between the edges
        bin_centers = np.zeros(num_bins)
        for k in range(1, num_bins + 1):
            bin_centers[k - 1] = (bin_edges[k] - bin_edges[k - 1]) / 2 + bin_edges[
                k - 1
            ]

        # Should return: xarray with name 'burst_counts,' and axis labels 'stimulus_presentation_id',
        # 'time_relative_to_stimulus_onset', and 'unit_id'
        return xarray.DataArray(
            data=presentationwise_counts,
            dims=[
                "stimulus_presentation_id",
                "time_relative_to_stimulus_onset",
                "unit_id",
            ],
            coords=dict(
                stimulus_presentation_id=(
                    ["stimulus_presentation_id"],
                    stimulus_presentation_ids,
                ),
                time_relative_to_stimulus_onset=(
                    ["time_relative_to_stimulus_onset"],
                    bin_centers,
                ),
                unit_id=(["unit_id"], np.int64(unit_ids)),
            ),
            name="burst_counts",
        )

    def presentationwise_burst_times(self, name, stimulus_presentation_ids, unit_ids):
        # TODO: Clarify in the documentation of this function (and the non_burst version)
        # that there is no shuffling supported by this function at all
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
        # for unit_id, bursts in burst_dict.items():
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
                # stim_presentation_ids = bursts["stimulus_presentation_id"]
                # unit id needs to be included. `bursts` is organized by unit,
                # so we just need to add a column full of this particular unit id
                unit_id_arr = np.full(len(bursts), unit_id, dtype=int)

                # Collect everything into a dataframe
                current_unit_presenationwise_times = pd.DataFrame(
                    {
                        "absolute_beg_time": abs_beg_times,
                        "absolute_end_time": abs_end_times,
                        "relative_beg_time": rel_beg_times,
                        "relative_end_time": rel_end_times,
                        "stimulus_presentation_id": bursts[
                            "stimulus_presentation_id"
                        ].to_numpy(dtype=int),
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

        # Type the dataframe columns correctly
        presentationwise_times.unit_id = presentationwise_times.unit_id.astype(int)
        presentationwise_times.stimulus_presentation_id = presentationwise_times.stimulus_presentation_id.astype(
            int
        )

        return presentationwise_times

    def presentationwise_non_burst_counts(
        self, name, bin_edges, stimulus_presentation_ids, unit_ids
    ):
        if not name in self._decoders.keys():
            raise ValueError(
                f"{name} did not match the name of any decoders constructed by this object."
            )
        elif not self._decoders[name].has_singles():
            raise ValueError(
                f"{name} does not have any isolated single spikes to count."
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
            stim_presentation_id_offset = int(stim_presentation_ids[0])
            single_dict = self._decoders[name].whole_single_dict
            bins = bin_edges

            num_presentations, num_bins, num_units = x.shape
            if type(unit_ids) != np.ndarray:
                unit_ids = np.array(unit_ids)
            num_units = unit_ids.size

        # If this data is shuffled, this function has already been called on all the units and
        # stim presentation ids, so we can just grab the data from the decoder and return that
        # with just the specified stim_ids and unit_ids
        if self._decoders[name].is_shuffled():
            presentationwise_counts = copy.deepcopy(self._decoders[name].single_counts)
            presentationwise_counts = presentationwise_counts.sel(
                stimulus_presentation_id=stimulus_presentation_ids, unit_id=unit_ids
            )
            return presentationwise_counts

        # Initialize the return array
        presentationwise_counts = np.zeros(
            (num_presentations, num_bins, num_units), dtype=int
        )

        # Walk through each bin, and check if any of the spike times are in it. If they are,
        # presentationwise_counts[presentation_id, bin_num, unit_id] += 1
        unit_idx = 0
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
                        (
                            (singles["relative_spike_time"] < next_bin)
                            * (singles["relative_spike_time"] >= current_bin)
                        )
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

                # Repeat the loop action for the last bin
                binned_bursts = singles.loc[(singles["relative_spike_time"] > next_bin)]
                for presentation_id in binned_bursts["stimulus_presentation_id"]:
                    presentation_idx = int(
                        presentation_id - stim_presentation_id_offset
                    )
                    presentationwise_counts[
                        presentation_idx, num_bins - 1, unit_idx
                    ] += 1
            unit_idx += 1

        # Take out any stim presentations not in stim_presentation_ids
        indices_to_keep = (
            np.array(stimulus_presentation_ids) - stim_presentation_id_offset
        )
        presentationwise_counts = presentationwise_counts[indices_to_keep]

        # Adjust the bin labels to be the center between the edges
        bin_centers = np.zeros(num_bins)
        for k in range(1, num_bins + 1):
            bin_centers[k - 1] = (bin_edges[k] - bin_edges[k - 1]) / 2 + bin_edges[
                k - 1
            ]

        # Should return: xarray with name 'single_spike_counts,' and axis labels 'stimulus_presentation_id',
        # 'time_relative_to_stimulus_onset', and 'unit_id'
        return xarray.DataArray(
            data=presentationwise_counts,
            dims=[
                "stimulus_presentation_id",
                "time_relative_to_stimulus_onset",
                "unit_id",
            ],
            coords=dict(
                stimulus_presentation_id=(
                    ["stimulus_presentation_id"],
                    stimulus_presentation_ids,
                ),
                time_relative_to_stimulus_onset=(
                    ["time_relative_to_stimulus_onset"],
                    bin_centers,
                ),
                unit_id=(["unit_id"], np.int64(unit_ids)),
            ),
            name="single_spike_counts",
        )

    def presentationwise_non_burst_times(
        self, name, stimulus_presentation_ids, unit_ids
    ):
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
        # for unit_id, singles in single_dict.items():
        for unit_id in unit_ids:
            singles = single_dict[unit_id]
            if singles is not None:
                # Only check the specified stimuli (specified by stimulus_presentation_ids)
                single_locations = singles.isin(stim_id_mask)[
                    "stimulus_presentation_id"
                ]
                singles = singles.loc[single_locations]

                # Get the columns from `singles` that describe the presentationwise burst times
                abs_spike_times = singles["absolute_spike_time"]
                rel_spike_times = singles["relative_spike_time"]
                stimulus_presentation_ids = singles["stimulus_presentation_id"]
                # unit id needs to be included. `singles` is organized by unit,
                # so we just need to add a column full of this particular unit id
                unit_id_arr = np.full(len(singles), unit_id)

                # Collect everything into a dataframe
                current_unit_presenationwise_times = pd.DataFrame(
                    {
                        "absolute_spike_time": abs_spike_times,
                        "relative_spike_time": rel_spike_times,
                        "stimulus_presentation_id": singles["stimulus_presentation_id"],
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
            by=["absolute_spike_time"], ascending=True, ignore_index=True
        )

        # Type the dataframe columns correctly
        presentationwise_times.unit_id = presentationwise_times.unit_id.astype(int)
        presentationwise_times.stimulus_presentation_id = presentationwise_times.stimulus_presentation_id.astype(
            int
        )

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
    def _shuffle_trials(
        self,
        bin_edges,
        stim_ids,
        stim_presentation_order,
        stim_classes,
        name,
        presentationwise_function,
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
        # For each stimulus condition (e.g. each presentation angle)
        for stim_class in stim_classes:
            class_presentation_indicies = []

            # Used later when collecting shuffled presentations in order
            counts[stim_class] = 0

            # Collect the indicies for each presentation of that class
            for k in range(num_presentations):
                if stim_class == stim_presentation_order[k]:
                    class_presentation_indicies = class_presentation_indicies + [k]

            # Get all of the activity counts for this stimulus condition
            presentations_by_class[stim_class] = np.array(
                presentationwise_function(
                    name,
                    bin_edges,
                    stim_ids[class_presentation_indicies],
                    self.all_units,
                )
            )

        # PSTHS: (num_presentations, num_bins, num_units)
        # We want to loop through each bin -> current_bin: (num_presentations, num_units)
        # Shuffle each column of current_bin (keeps units' responses within unit, but outside of trial)
        for stim_class in stim_classes:
            # current_class_psth = presentations_by_class[stim_class]
            for bin_idx in range(num_bins):
                for unit_idx in range(num_units):
                    rng.shuffle(
                        presentations_by_class[stim_class][:, bin_idx, unit_idx]
                    )
                # current_bin = current_class_psth[:,bin_idx,:]
                # Shuffle the current bin column wise
                # current_class_psth[:,bin_idx,:] = rng.shuffle(current_bin, axis=1)

        # Now all the stimulus presentations are shuffled within class, and we need to collect them
        # all into one array (with the original class presentation ordering given by
        # stim_presentation_order)
        presentation_idx = 0
        psths = np.zeros((num_presentations, num_bins, num_units), dtype=int)
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

        # Adjust the bin labels to be the center between the edges
        bin_centers = np.zeros(num_bins)
        for k in range(1, num_bins + 1):
            bin_centers[k - 1] = (bin_edges[k] - bin_edges[k - 1]) / 2 + bin_edges[
                k - 1
            ]

        # return psths
        return xarray.DataArray(
            data=psths,
            dims=[
                "stimulus_presentation_id",
                "time_relative_to_stimulus_onset",
                "unit_id",
            ],
            coords=dict(
                stimulus_presentation_id=(["stimulus_presentation_id"], stim_ids),
                time_relative_to_stimulus_onset=(
                    ["time_relative_to_stimulus_onset"],
                    bin_centers,
                ),
                unit_id=(["unit_id"], np.int64(self.all_units)),
            ),
            # name="burst_counts",
        )

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
        if bin_stop == 0.0:
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

    def _check_membership(self, members, arr_to_check):
        arr_mask = np.zeros(arr_to_check.shape, dtype=bool)
        for k in range(len(arr_mask)):
            arr_mask[k] = arr_to_check[k] in members
        return arr_mask
