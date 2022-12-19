from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import callbacks, models
import tensorflow as tf
import numpy as np
import random
import os
import datetime

from agent import Agent

GAMMA = 0.95
MEMORY_SIZE = 1000000
LEARNING_RATE = 0.001
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
BATCH_SIZE = 20
CHECKPOINT_RATE = 0.1

class DQNAgent(Agent):
	def __init__(self, env):
		super().__init__(env)
		self.observation_space = env.observation_space.shape[0]
		self.action_space = env.action_space.n
		self.memory = deque(maxlen=MEMORY_SIZE)
		self.exploration_rate = EXPLORATION_MAX

		# self.checkpoint_dir = "./ckpt"
		# if not os.path.exists(self.checkpoint_dir):
		# 	os.makedirs(self.checkpoint_dir)

		self._build_model()

	def _build_model(self):
		self.model = Sequential()
		self.model.add(Input(shape=(4,1)))
		self.model.add(Dense(128, activation="relu"))
		# self.model.add(Dense(128, input_shape=(4,1), activation="relu"))
		self.model.add(Dense(self.action_space, activation="linear"))
		self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))


	def save_observation(self, observation, action, reward, next_observation, done):
		self.memory.append((observation, action, reward, next_observation, done))

	def uprank_prediction_data(self, observation):
		# This class was inspired by some old code. This is a patch for breaking
		# changes in the keras APIs. See:
		# https://github.com/mrdbourke/tensorflow-deep-learning/discussions/278
		return tf.expand_dims(np.array(observation), axis=-1)

	def action(self, observation):
		if np.random.rand() < self.exploration_rate:
			return random.randrange(self.action_space)
		
		q_values = self.model.predict(self.uprank_prediction_data(observation), verbose=0)
		return np.argmax(q_values[0])


	def train(self):
		if len(self.memory) < BATCH_SIZE:
			return

		batch = random.sample(self.memory, BATCH_SIZE)
		for observation, action, reward, next_observation, done in batch:
			q_update = reward
			prediction_input = self.uprank_prediction_data(next_observation)
			if not done:
				q_update = (reward + GAMMA * np.amax(self.model.predict(prediction_input, verbose=0)))
			q_values = self.model.predict(prediction_input, verbose=0)
			q_values[0][0][action] = q_update
			
			# checkpoint_callback = [
      #   # This callback saves a SavedModel every epoch
      #   # We include the current epoch in the folder name.
      #   callbacks.ModelCheckpoint(
      #       filepath=self.checkpoint_dir + f"/ckpt-{datetime.datetime.now().isoformat()}"
      #   )
			# ]
			# checkpoints = [self.checkpoint_dir + "/" + name for name in os.listdir(self.checkpoint_dir)]

			# self.model.fit(prediction_input, q_values, verbose=0, callbacks=checkpoint_callback)
			self.model.fit(prediction_input, q_values, verbose=0)

		self.exploration_rate *= EXPLORATION_DECAY
		self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
