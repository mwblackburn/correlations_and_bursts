import os
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
        pass

    def load_session(self, session_id):
        # Out: session object
        pass

    def get_session_table(self):
        return self._session_table
