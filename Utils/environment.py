import gym
import collections
import cv2
import numpy as np


class MaxAndSkipEnv(gym.Wrapper):
    """
    MaxAndSkipEnv is a wrapper that modifies the behavior of the Gym environment. 
    It returns only every `skip`-th frame during the step method call, and takes the element-wise maximum
    over the skipped frames (max pooling). This allows for faster action approximation in situations 
    where consecutive frames might be similar.
    """
    def __init__(self, enviroment=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(enviroment)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        for _ in range(self._skip):
            observation, reward, done, info = self.env.step(action)
            self._obs_buffer.append(observation)
            total_reward += reward
            if done:
                break
        # Max pool over the last two observations
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init to first obs"""
        self._obs_buffer.clear()
        observation = self.env.reset()
        self._obs_buffer.append(observation)
        return observation


class ProcessFrame84(gym.ObservationWrapper):
    """
    ProcessFrame84 is a Gym environment wrapper that processes raw observations 
    by downsampling them to an 84x84 pixel resolution and converting them to grayscale.
    
    Downsamples image to 84x84
    Greyscales image

    Returns numpy array
    """

    def __init__(self, enviroment=None):
        super(ProcessFrame84, self).__init__(enviroment)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, observation):
        return ProcessFrame84.process(observation)

    @staticmethod
    def process(frame):
        """
           Static method that takes a raw frame as input and processes it to reduce
           it to an 84x84 pixel resolution.
           It performs weighted grayscale conversion (luminance), then resizes 
           the image to 84x110 pixels and crops the central 84x84 part.
           Returns the processed image as a numpy array with shape (84, 84, 1) and type np.uint8.
        """
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        
        # Weighted grayscale conversion (luminance)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        
        # Resize and crop
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    """
    This wrapper modifies the environment observations to suit PyTorch.
    It changes the observation space shape to (channel, height, width) 
    and the observation method moves the axes accordingly.
    """
    def __init__(self, enviroment):
        super(ImageToPyTorch, self).__init__(enviroment)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        """Moves the channel axis to the first dimension (channel, height, width)."""
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    This wrapper normalizes pixel values of the observation to the range from 0 to 1.
    This is used for QL/SARSA.
    """
    def observation(self, observation):
        """Divides each value by 255.0."""
        return np.array(observation).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    """
    This wrapper creates an observation buffer to keep track of the last n_steps frames 
    (used for stacking frames, e.g., 4 frames).
    """
    def __init__(self, enviroment, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(enviroment)
        self.buffer = None
        self.dtype = dtype
        old_space = enviroment.observation_space
        # Update observation space to include the stacked frames
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        """Resets the buffer and returns the initial observation of the environment."""
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """Updates the buffer by shifting elements and inserting the new observation."""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class PixelNormalization(gym.ObservationWrapper):
    """
    Normalize pixel values in frame --> 0 to 1
    This is used for DDQN.
    """
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0