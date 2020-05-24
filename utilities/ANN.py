import faiss
import pyflann



class NNSearcher:
    # TODO: Optimize this searcher, faiss seems to be much quicker, we should experiment with the different methods.
    
    def __init__(self, database_features, anntype='faiss', useGpu=False): 
        assert anntype in ['faiss','flann'] 
        self.anntype = anntype

        if anntype == 'faiss': 
            # Making the SCCvSD edge images searchable
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(2016)

            if useGpu == True:
                nnsearcher = faiss.index_cpu_to_gpu(res, 0, index)
            else:
                nnsearcher = index
            # nnsearcher = faiss.IndexFlatIP(2016) this seems to be slower..

            # Add representations to the database
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


