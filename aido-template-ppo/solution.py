#!/usr/bin/env python3

import io

import numpy as np
from PIL import Image

from aido_schemas import (
    Context,
    DB20Commands,
    DB20Observations,
    EpisodeStart,
    JPGImage,
    LEDSCommands,
    logger,
    protocol_agent_DB20,
    PWMCommands,
    RGB,
    wrap_direct,
)

from agent import Agent


def velangle_to_lrpower(action, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0):
    vel, angle = action

    # Distance between the wheels
    baseline = 0.102

    # assuming same motor constants k for both motors
    k_r = k
    k_l = k

    # adjusting k by gain and trim
    k_r_inv = (gain + trim) / k_r
    k_l_inv = (gain - trim) / k_l

    omega_r = (vel + 0.5 * angle * baseline) / radius
    omega_l = (vel - 0.5 * angle * baseline) / radius

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv

    # limiting output to limit, which is 1.0 for the duckiebot
    u_r_limited = max(min(u_r, limit), -limit)
    u_l_limited = max(min(u_l, limit), -limit)

    vels = np.array([u_l_limited, u_r_limited])

    return vels


class PPOAgent:
    n: int

    def init(self, context: Context):
        self.stack = []
        context.info("init()")

        self.agent = Agent()
        # self.agent.load_param(filename="ppo_net_params_zigzag.pkl")
        # self.agent.load_param(filename="checkpoint.pkl")
        # self.agent.load_param(filename="ppo_net_params_Czigzag.pkl")
        self.agent.load_param(filename="ppo_net_params_Cdir_slower_no_speed_reward_lowlr.pkl")
        self.command = [0, 0]

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, context: Context, data: DB20Observations):
        logger.info("received", data=data)
        camera: JPGImage = data.camera
        odometry = data.odometry
        print(odometry)
        img_gray = jpg2rgb(camera.jpg_data)

        if len(self.stack) == 0:
            self.stack = [img_gray] * 4
        else:
            self.stack.pop(0)
            self.stack.append(img_gray)



        velangle, alogp = self.agent.select_action(np.array(self.stack))
        # velangle = [velangle[0], .25] # fix speed
        velangle = velangle*np.array([1., 2.]) + np.array([0., -1.]) # Scale to proper range

        # hack speed
        # velangle[1] = np.clip(velangle[1] * 2, -1, 1)
        velangle[0] = max(.501 - np.abs(velangle[1]), .1)

        # velangle *= 1.5
        # velangle[0] = max(velangle[0], 0.2)

        self.command = velangle_to_lrpower(velangle)
        # self.command = (lrpower * 2 - 1) * .5

    def on_received_get_commands(self, context: Context):
        pwm_left, pwm_right = self.command

        # pwm_left = 1.0
        # pwm_right = 1.0
        col = RGB(0.0, 0.0, 1.0)
        led_commands = LEDSCommands(col, col, col, col, col)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write("commands", commands)

    def finish(self, context: Context):
        context.info("finish()")


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""

    im = Image.open(io.BytesIO(image_data))
    im = im.convert("RGB")
    im = im.resize((96, 96))
    data = np.array(im)

    data = rgb2gray(data)

    return data


def rgb2gray(rgb, norm=True):
    # rgb image -> gray [0, 1]
    gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
    if norm:
        # normalize
        gray = gray / 128. - 1.
    return gray


def main():
    node = PPOAgent()
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == "__main__":
    main()

# conda activate gym-duckietown
# usermod -a -G docker thomas
# newgrp docker
# dts challenges evaluate --challenge aido5-LF-sim-validation


# mpv /LFv-sim/challenge-evaluation-output/episodes/ETHZ_autolab_technical_track-sc*/ego/camera.mp4

#random score
# driven_any_mean: 1.8510551780477624
# survival_time_mean: 4.437499999999992

# PPO
# survival_time_mean: 12.100000000000058
# driven_any_mean: 3.9647670197931304