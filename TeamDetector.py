
from sklearn.cluster import KMeans
from typing import List
from Team import Team
import numpy as np

class TeamDetector: 
    def __init__(
            self, 
            teams: List[Team]
            ): 
        self.teams = teams
        n_clusters = len(teams)
        jersey_colors = np.vstack([t.jersey_color for t in teams])
        self.predictor = KMeans(n_clusters=n_clusters, random_state=0, init=jersey_colors)

    def get_teams(self, tracked_persons):
        colors = None
        for tp in tracked_persons: 
            assert tp.color.shape[0] == 3 
            if colors is None : 
                colors = tp.color
            else: 
                colors = np.vstack([colors,tp.color])
        self.predictor.fit(colors)
        clusters = self.predictor.predict(colors)
        teams = [Team._teams[c] for c in clusters ]

        return teams
