from _commons import warn, error, create_dir_path
import numpy as np

class LinUCB:
    def __init__(self, alpha, dataset, max_items=500):
        self.dataset = dataset
        self.alpha = alpha
        self.d = dataset.arm_feature_dim
        self.b = np.zeros(shape=(dataset.num_items, self.d))

        self.A = np.zeros(shape=(dataset.num_items, self.d, self.d)) # set of arms is not changing over time
        #for a in range(self.A.shape[0]):
        #    self.A[a] = np.identity(self.d, dtype=self.A.dtype)

        # More efficient way to create array of identity matrices of length num_items
        print("Initializing matrix A of shape {} which will require {}MB of memory.".format(self.A.shape, 8*self.A.size/1e6))
        self.A = np.tile(np.identity(self.d, dtype=self.A.dtype), (dataset.num_items, 1))
        print("Initialized matrix A of shape", self.A.shape)

    def choose_arm(self, t):
        A = self.A
        b = self.b
        arm_features = self.dataset.get_features_of_current_arms(t=t)
        p_t = np.zeros(shape=(arm_features.shape[0],), dtype=float)
        for a in range(arm_features.shape[0]):
            x_ta = arm_features[a]
            A_a_inv = np.linalg.inv(A[a])
            theta_a = A_a_inv.dot(b[a])
            p_t[a] = theta_a.T.dot(x_ta) + self.alpha*np.sqrt(x_ta.T.dot(A_a_inv).dot(x_ta))

        max_idxs = np.argwhere(p_t == np.max(p_t)) #I want to randomly break ties, np.argmax return the first occurence of maximum.
        a_t = np.random.choice(max_idxs) #idx of article to recommend to user t

        r_t = self.dataset.recommend(user_id=t, item_id=a_t) # observed reward = 1/0

        x_t_at = arm_features[a_t]
        A[a_t] = A[a_t] + x_t_at.dot(x_t_at.T)
        b[a_t] = b[a_t] + r_t*x_t_at