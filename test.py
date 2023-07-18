import time
import gym
frame_time = 1.0 / 15 # seconds
n_episodes = 1
env=gym.make('MsPacman-v0')
for i_episode in range(n_episodes):
    t=0
    score=0
    then = 0
    done = False
    env.reset()
    while not done:
        now = time.time()
        if frame_time < now - then:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            score += reward
            env.render()
            then = now
            t=t+1
    print('Episode {} finished at t {} with score {}'.format(i_episode,
                                                             t,score))