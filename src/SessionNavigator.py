import os
import copy
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Include functions for finding sessions, loading them
# Data wrangler
# Descriptive statistics module (class), correlations
# Decoding class
# Plotting/visualization class


class SessionNavigator:

    # TODO: Doc me

    def __init__(self, manifest_path):
        if not (os.path.isdir(manifest_path)):
            raise FileNotFoundError("The provided manifest_path does not exist")

        self._working_directory = os.getcwd()
        self._cache = EcephysProjectCache(manifest=manifest_path)
        self._session_table = self._cache.get_session_table()

    def find_sessions(self, visual_areas=[], genotype="", session_type=""):
        # in: criteria
        # out: sessionID numbers that meet the criteria
        sessions = copy.deepcopy(self._session_table)

        # Empty strings and lists evaluate as false, so if the argument was provided,
        # these ifs will evaluate to true
        if session_type:
            sessions = sessions[(sessions.session_type == session_type)]

        if genotype:
            sessions = sessions[(sessions.full_genotype.str.find(genotype) > -1)]

        if visual_areas:
            for area in visual_areas:
                sessions = sessions[
                    (
                        [
                            area in acronyms
                            for acronyms in sessions.ecephys_structure_acronyms
                        ]
                    )
                ]

        return sessions.index.values

    def load_session(self, session_id):
        # Out: session object
        return self._cache.get_session_data(session_id=session_id)

    def get_session_table(self):
        return self._session_table