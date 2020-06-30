
import numpy as np 

class Team:
    _teams = {}
    def __init__(self, 
        id: int, 
        name: str,
        jersey_color: np.array
    ):
        self.id = id
        self.name = name 
        assert jersey_color.shape[0] == 3 # Should be a 3-channel color
        self.jersey_color = jersey_color
        Team._teams[self.id] = self
        
    def get_name(self):
        return self.name
    
    @classmethod
    def get_team(cls, id): 
        return cls._teams[id]