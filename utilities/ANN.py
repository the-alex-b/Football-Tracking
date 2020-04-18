import faiss
import pyflann



class NNSearcher:
    
    def __init__(self, database_features, anntype='faiss'): 
        assert anntype in ['faiss','flann'] 
        self.anntype = anntype

        if anntype == 'faiss': 
            # Making the SCCvSD edge images searchable
            nnsearcher = faiss.IndexFlatL2(2016)
            nnsearcher.add(database_features.copy())
        elif anntype == 'flann': 
            # Initialize a flann
            nnsearcher = pyflann.FLANN()
            self.database_features = database_features

        self.nnsearcher = nnsearcher
        

    def seek_nn(self, features):
        if self.anntype == 'faiss': 
            _, retrieved_index = self.nnsearcher.search(features.copy(), 1)
            retrieved_index = retrieved_index[:,0][0]
            return retrieved_index
        elif self.anntype == 'flann': 
            result, _ = self.nnsearcher.nn(self.database_features, features, 1, algorithm="kdtree", trees=16, checks=64)
            retrieved_index = result[0]
            return retrieved_index


