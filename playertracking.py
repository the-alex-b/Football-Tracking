
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances as euc
from sklearn.metrics.pairwise import cosine_distances as cosd
import numpy as np
import cv2

def get_flow_keypoints(tar_frame,direction,nskip,frames,gap=1):
  if direction == 'fwd':
    f_o_b = -1
  if direction == 'bwd':
    f_o_b = 1
  old_gray = cv2.cvtColor(frames[tar_frame + nskip * f_o_b].frame,cv2.COLOR_RGB2GRAY)

  lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  
  ndetect = frames[tar_frame + nskip * f_o_b].playerkeypoints.shape[0]
  nkeyps = frames[tar_frame + nskip * f_o_b].playerkeypoints.shape[1]
  p0 = np.float32(frames[tar_frame + nskip * f_o_b].playerkeypoints.reshape((ndetect*nkeyps,3))[:,0:2][:,None,:])

  for j in range(nskip * gap):
      frame = frames[-f_o_b*(j + 1) + gap*(tar_frame + nskip * f_o_b)].frame
      frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # calculate optical flow
      p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
      old_gray = frame_gray.copy()
      p0 = p1

  p1 = p1.reshape((ndetect,nkeyps,2))
  return p1


def reassign(trees,res,frame,det,radius,unique_objs,k_max):
  n_flow = len(trees)
  obj = []
  dis = []
  for j in range(n_flow):
    tree = trees[j]
    candidates = tree.radius_neighbors(res[frame][det,0:2].reshape(1, -1),radius = radius)
    candidate_objs = candidates[1][0]
    candidate_dis = candidates[0][0]
    candidate_objs = res[frame - j - 1][candidate_objs,2]
    is_inactive = [not(cand in unique_objs) for cand in candidate_objs]
    if len(is_inactive) > 0:
      obj.extend(candidate_objs[is_inactive])
      dis.extend(candidate_dis[is_inactive])
  if len(obj) > 0:
    new_id = np.argmin(np.array(dis))
    new_id = obj[new_id]
    created = 0
  else:
    new_id = k_max + 1
    created = 1
  return new_id, created
      

##### Main #####

def track(frames): 

    n_det_frames = len(frames)

    lamb = 10
    Nq = 8
    Ktau = 3
    radius = 50

    Q = lamb / Nq
    tau = (Nq * (Ktau - 1) + 1) / (Nq - 1)
    max_n_flow = 8
    frame_idx = 0
    res = [None] * n_det_frames
    #det_centroids = np.mean(results[frame_idx]['keypoints'][...,0:2],axis = 1)
    det_centroids = frames[frame_idx].playersneckcoos[...,0:2] 
    res[frame_idx] = np.c_[det_centroids,np.array(range(det_centroids.shape[0]))]
    k_max = det_centroids.shape[0]

    while frame_idx < n_det_frames-1:
        frame_idx += 1
        #det_centroids = np.mean(results[frame_idx]['keypoints'][...,0:2],axis = 1)
        det_centroids = frames[frame_idx].playersneckcoos[...,0:2]
        n_obj = det_centroids.shape[0]
        n_flow = min(max_n_flow,frame_idx)
        opt_flow_centroids = [None] * n_flow
        trees = [None] * n_flow
        indices = np.empty((n_flow + 1,n_obj))
        indices[n_flow,...] = -1
        distances = np.empty((n_flow + 1,n_obj))
        for j in range(n_flow):
            opt_flow_centroids[j] = np.mean(get_flow_keypoints(frame_idx,'fwd',j+1,frames),axis = 1)
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(opt_flow_centroids[j])
            trees[j] = nbrs
            dists, inds = nbrs.kneighbors(det_centroids)
            distances[j,...] = dists[...,0]
            indices[j,...] = inds[...,0]
            distances[j,...] = j * Q + distances[j] / (1 + j * tau)
        distances[n_flow,...] = lamb
        winners = np.argmin(distances,axis = 0) # which prediction is closest to the observation?
        res[frame_idx] = np.empty((n_obj,3))
        res[frame_idx][:,0:2] = det_centroids
        for j in range(n_obj):
            if winners[j] != n_flow:
                res[frame_idx][j,2] = res[frame_idx - winners[j] - 1][int(indices[winners[j],j]),2]
            else:
                res[frame_idx][j,2] = k_max + 1
                k_max += 1

        unique_objs = np.unique(res[frame_idx][:,2])
        assignments = [np.where(res[frame_idx][:,2] == unique_objs[j])[0]  for j in range(unique_objs.size)]
        n_per_obj = [assignments[j].size for j in range(len(assignments))]
        duped = np.where(np.array(n_per_obj) > 1)[0]
        
        for dupe in list(duped):
            query_pts = frames[frame_idx].playergeneralfeatures[assignments[dupe],:]
            ref_pts = np.vstack(tuple([frames[j].playergeneralfeatures[res[j][:,2] == unique_objs[dupe]] for j in range(frame_idx)]))
            winner = np.argmin(np.min(cosd(query_pts,ref_pts),axis = 1))
            detections_to_reassign = np.delete(assignments[dupe],winner)
            for det in detections_to_reassign:
                new_id,created = reassign(trees,res,frame_idx,det,radius,unique_objs,k_max)
                res[frame_idx][det,2] = new_id
                k_max = k_max + created

    return res


