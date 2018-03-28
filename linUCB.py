from _commons import warn, error, create_dir_path
import numpy as np
import time


class LinUCB:
    def __init__(self, alpha, dataset, max_items=500):
        self.dataset = dataset
        self.dataset.shrink(max_items)
        self.dataset.add_random_ratings(num_to_each_user=5)
        self.alpha = alpha
        self.d = dataset.arm_feature_dim
        self.b = np.zeros(shape=(dataset.num_items, self.d))

        self.A = np.zeros(shape=(dataset.num_items, self.d, self.d)) # set of arms is not changing over time
        #for a in range(self.A.shape[0]):
        #    self.A[a] = np.identity(self.d, dtype=self.A.dtype)

        # More efficient way to create array of identity matrices of length num_items
        print("Initializing matrix A of shape {} which will require {}MB of memory.".format(self.A.shape,
                                                                                            8 * self.A.size / 1e6))
        self.A = np.repeat(np.identity(self.d, dtype=self.A.dtype)[np.newaxis, :, :], dataset.num_items, axis=0)
        print("\nLinUCB successfully initialized.")

    def choose_arm(self, t):
        """
        Choose an arm to pull = item to recommend to user t.
        :param t: User_id of user to recommend to.
        :return: Received reward for selected item = 1/0 = user liked/disliked item.
        """
        A = self.A
        b = self.b
        arm_features = self.dataset.get_features_of_current_arms(t=t)
        p_t = np.zeros(shape=(arm_features.shape[0],), dtype=float)
        for a in range(arm_features.shape[0]):
            x_ta = arm_features[a]
            A_a_inv = np.linalg.inv(A[a])
            theta_a = A_a_inv.dot(b[a])
            p_t[a] = theta_a.T.dot(x_ta) + self.alpha * np.sqrt(x_ta.T.dot(A_a_inv).dot(x_ta))

        # I want to randomly break ties, np.argmax return the first occurence of maximum.
        max_idxs = np.argwhere(p_t == np.max(p_t)).flatten()
        a_t = np.random.choice(max_idxs)  # idx of article to recommend to user t

        r_t = self.dataset.recommend(user_id=t, item_id=a_t)  # observed reward = 1/0

        x_t_at = arm_features[a_t]
        A[a_t] = A[a_t] + x_t_at.dot(x_t_at.T)
        b[a_t] = b[a_t] + r_t * x_t_at

        return r_t

    def run_epoch(self):
        """
        Call choose_arm() for each user in the dataset.
        :return: Average received reward.
        """
        rewards = np.zeros(shape=(self.dataset.num_users,), dtype=float)
        start_time = time.time()
        for i in range(self.dataset.num_users):
            start_time_i = time.time()
            user_id = self.dataset.get_next_user()
            rewards[i] = self.choose_arm(user_id)
            time_i = time.time() - start_time_i
            print("Choosing arm for user {}/{} ended with reward {} in {}s".format(i, self.dataset.num_users, rewards[i], time_i))

        total_time = time.time() - start_time
        avg_reward = np.average(rewards)
        return avg_reward, total_time

    def run(self, num_epochs):
        """
        Runs run_epoch() num_epoch times.
        :param num_epochs: Number of epochs = iterating over all users.
        :return: List of average rewards per epoch.
        """
        avg_rewards = np.zeros(shape=(num_epochs,), dtype=float)
        for i in range(num_epochs):
            avg_rewards[i], total_time = self.run_epoch()

            print("Finished epoch {}/{} with avg reward {} in {}s".format(i, num_epochs, avg_rewards[i], total_time))
        return avg_rewards