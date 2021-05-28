import gfootball.env as football_env
import torch
render = False
seed = 1

def main():
    env = football_env.create_environment(
      env_name='academy_3_vs_1_with_keeper',
      stacked=False,
      representation='simple115v2',
      rewards = 'scoring',
      write_goal_dumps=False,
      write_full_episode_dumps=False,
      dump_frequency = 0)
    win=0
    lost=0
    torch.manual_seed(seed)
    env.seed(seed)
    for i_epoch in range(1000000):
        
        #print(seed)
        env.reset()
        step = 0
        sum = 0.0   
        while True:
            step+=1
            if render: env.render()
            if step==0:
                action=13
            if step>=0 and step<10:
                action=1
            if step>=10 and step<50:
                action=7
            if step>=50 and step<70:
                action=5
            if step>=70 and step<95:
                action=4
            if step==94:
                action=15
            if step==95:
                action=12
            next_state, reward, done, _ = env.step(action)
            sum+=reward
            if done:
                break
        if reward==1: win+=1
        if reward==-1:lost+=1
        print("{}:{}\t\tWin:{}\t\tLost:{}".format(i_epoch,sum,win,lost))

if __name__ == '__main__':
    main()
    
    print("end")



