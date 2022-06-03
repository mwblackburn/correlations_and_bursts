import os
import copy
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


class SessionNavigator:

    # TODO: Doc me

    def __init__(self, manifest_path):
        if not (os.path.isfile(manifest_path)):
            raise FileNotFoundError("The provided manifest_path does not exist.")

        self._working_directory = os.getcwd()
        self._cache = EcephysProjectCache(manifest=manifest_path)
        self._session_table = self._cache.get_session_table()
        self._debug = True
        return

    def find_sessions(self, visual_areas=[], genotype="", session_type=""):
        # in: criteria
        # out: sessionID numbers that meet the criteria
        sessions = copy.deepcopy(self._session_table)
        # aaa = sessions.full_genotype.values

        # Empty strings and lists evaluate as false, so if the argument was provided,
        # these ifs will evaluate to true
        if session_type:
            if session_type not in sessions.session_type.values:
                raise ValueError(f"{session_type} is not a valid session type.")
            sessions = sessions[(sessions.session_type == session_type)]

        if genotype:
            if genotype not in sessions.full_genotype.values:
                raise ValueError(f"{genotype} is not a valid genotype.")
            # sessions = sessions[(sessions.full_genotype.str.find(genotype) > -1)]
            sessions = sessions[(sessions.full_genotype.values == genotype)]

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
