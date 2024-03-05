import gym
import imageio
from PIL import Image, ImageDraw
import numpy as np
import torch
import sys
import os
from gym import spaces

# os.environ["MUJOCO_GL"] = "egl"

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)
# sys.path.append(parent_dir+"/onpolicy/algorithm")
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.envs.IA.IA_env import IAEnv
from onpolicy.config import get_config

def _t2n(x):
    return x.detach().cpu().numpy()

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='ia_simple', help="Which scenario to run on")
    parser.add_argument("--num_harvester", type=int, default=3, help="number of harvesters")
    parser.add_argument('--num_transporter', type=int,
                        default=2, help="number of transporters")
    
    # environment settings
    parser.add_argument('--dt', type=float, default=0.1, help="simulation interval")
    
    # Field settings
    parser.add_argument("--field_width", type=float, default=100.0, help="width of the farmland")
    parser.add_argument("--field_length", type=float, default=500.0, help="length of the farmland")
    parser.add_argument("--working_width", type=float, default=10.0, help="working width of the harvester")
    parser.add_argument("--headland_width", type=float, default=3.0, help="width of the headland")
    parser.add_argument("--ridge_width", type=float, default=0.3, help="width of the ridge")

    # Harvester parameters
    parser.add_argument("--yeild_per_meter", type=float, default=1.5, help="yeild of the harvester per meter")
    parser.add_argument("--transporting_speed", type=float, default=10.0, help="transporting speed of the harvester per second")
    parser.add_argument("--harvester_max_v", nargs='+', help='<Required> The maximum velocity of each harvester', required=True)
    parser.add_argument("--cap_harvester", nargs='+', help='<Required> The load capacity of each harvester', required=True) 

    # Transpoter parameters
    parser.add_argument("--unloading_speed", type=float, default=30.0, help="unloading speed of the transporter to the depot per second")
    parser.add_argument("--transporter_max_v", nargs='+', help='<Required> The maximum velocity of each transporter', required=True)  
    parser.add_argument("--cap_transporter", nargs='+', help='<Required> The load capacity of each harvester', required=True)
    
    # Other settings
    parser.add_argument("--shared_reward", action='store_false', default=True, help='Whether agent share the same rewadr')
    parser.add_argument("--d_range", type=float, default=1.0, help="the distance between the harv and trans to transporting")

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    env = IAEnv(all_args)
    eval_episode_rewards = []
    actors = []
    obs = env.reset()

    eval_rnn_states = np.zeros((1),dtype=np.float32)
    eval_masks = np.ones((1), dtype=np.float32)

    for i in range(all_args.num_transporter):
        act = R_Actor(all_args,env.observation_space[i],env.action_space[i])
        act.load_state_dict(torch.load("./onpolicy/scripts/results/IA/ia_simple/mappo/check/run1/models/actor.pt"))
        actors.append(act)
        # print(act.act.action_out.logstd._bias)

    with torch.no_grad():
        # print(env.env.initial_gripper1_pos,env.env.initial_gripper1_rot,env.env.initial_gripper2_pos,env.env.initial_gripper2_rot)
        for eval_step in range(all_args.episode_length):
            print("step: ", eval_step, "\n")
            env.render()
            action = []
            for agent_id in range(all_args.num_transporter):
                actor = actors[agent_id]
                actor.eval()
                # print(actor.act.action_out.log_std)
                print("agent id: ", agent_id, "observation: ", obs[agent_id], "\n")
                print("transporting: ", obs[agent_id][8], env.world.transporters[agent_id].transporting)
                eval_action,_,rnn_states_actor = actor(
                    obs[agent_id],
                    eval_rnn_states,
                    eval_masks,
                    deterministic=True,
                )
                eval_action = eval_action.detach().cpu().numpy()
                print("action: ",eval_action)
                action.append(eval_action)

            obs, eval_rewards, done, infos = env.step(np.stack(action).squeeze())
            print("reward: ",eval_rewards)
            # print("action: ",np.stack(action).squeeze().reshape(all_args.num_agents,3))
            eval_episode_rewards.append(eval_rewards)
        print("episode reward: ",np.array(eval_episode_rewards).sum())
        
        # writer = imageio.get_writer(parent_dir + "/render.gif")
        # # print('reward is {}'.format(self.reward_lst))
        # for frame, reward in zip(frames, eval_episode_rewards):
        #     print(eval_step)
        #     frame = Image.fromarray(frame)
        #     draw = ImageDraw.Draw(frame)
        #     draw.text((70, 70), '{}'.format(reward), fill=(255, 255, 255))
        #     frame = np.array(frame)
        #     writer.append_data(frame)
        # writer.close()
        # env.close()

if __name__ == "__main__":
    main(sys.argv[1:])