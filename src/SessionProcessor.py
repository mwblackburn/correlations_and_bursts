import numpy as np
from sklearn.svm import LinearSVC
from src.Decoder import Decoder

#import allensdk.brain_observatory.ecephys.ecephys_session

# This is where most of the calculations on a single session are going to be performed.
# It will:
# construct and evaluate decoders
# construct and evaluate PSTHs
# construct and evaluate correlation statistics
# save any relevant data for plots (so basically everything)
# 
# Maybe create a unit object that holds everything relevant for that unit?
# Maybe create a decoder object that holds the decoder, stim, bins, etc?
#   (actually a better idea would be to create a stim table object and 
#    store the decoder, bins, stim, etc. with it)

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
            current = list(session.units[session.units.ecephys_structure_acronym==acronym].index)
            
            self.units_by_acronym[acronym] = current
            self.all_units = self.all_units + current
            stop = len(self.all_units)
            self._all_unit_dividing_indices[acronym] = (start, stop)
        
        # Eventually this Processor is going to be decoding stimuli, so we'll need to store
        # the decoders
        self._decoders = {}
        
        #TODO: Construct more variables as needed
        

    # In theory, you could cause a bug here by calling this twice, with the same stim and bin
    # bin width, but different start and stop times. Be aware of this
    def construct_decoder(self, stim, name="", bin_start=0, bin_stop=0, bin_width=0.05, classifier=LinearSVC()):
        stim_table = self.session.get_stimulus_table(stim)
        stim_ids = stim_table.index.values
        
        if bin_stop == 0: bin_stop = stim_table.duration.mean() + bin_width
        bins = np.arrange(bin_start, bin_stop, bin_width)
        
        x = np.array(self.session.presentationwise_spike_counts(bins, stim_ids, self.all_units))
        y = np.array(stim_table[stim])
        
        # Change the null labels to a numerical value, so that the classifiers don't panic
        for idx in range(len(y)):
            if y[idx] == 'null':
                y[idx] = -1
        
        # If a name wasn't provide it, name it as explicitly as possible
        if name == "": name = f"{stim}_{bin_width*1000}ms"
        if name in self._decoders.keys(): 
            raise ValueError(f"A decoder with the name {name}, already exists. This can happen if you have a constructed a decoder with the same stimulus type and bin width.")
        
        self._decoders[name] = Decoder(classifier, stim, stim_table, bins, x, y, name=name)
        return # TODO: This function may not be complete
    
    # Idea: Eventually what we want, is to pass a set of stim names, and return psths, weights,
    # correlations, etc for every stim
    
    def construct_psth(self):
        pass
    
    def calculate_decoder_weights(self):
        pass
    
    def calculate_correlations(self):
        pass
    
    def save(self, path=''):
        pass