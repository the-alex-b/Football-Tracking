import pickle

'''
For now we use pickle to store the extracted frames on disk. Is this optimal? Or should we change it
'''
def write_extracted_frames_to_disk(extractedFrames):
    pickle.dump(extractedFrames, open("./storage/extracted_frames/latest_stored_extracted_frames.p", "wb"))



def load_extracted_frames_from_disk():
    storedExtractedFrames= pickle.load(open("./storage/extracted_frames/latest_stored_extracted_frames.p", "rb"))
    
    return storedExtractedFrames
