class argument():
    def __init__(self):
        self.GAMMA = 0.95
        self.EPOCHS = 5
        self.BATCH_SIZE = 64
        self.EPS_CLIP = 0.2
        self.ACTOR_LEARNING_RATE = 1e-5
        self.CRITIC_LEARNING_RATE = 1e-5
        self.TRAIN_EPISODE = 1000