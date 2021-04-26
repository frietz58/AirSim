from gym.envs.registration import register

register(
    id="airsim-drone-sample-v0", entry_point="airgym.envs:AirSimDroneEnv",
)

register(
    id="airsim-my-drone-env-v0", entry_point="airgym.envs:MyAirSimDroneEnv",
)

register(
    id="airsim-car-sample-v0", entry_point="airgym.envs:AirSimCarEnv",
)
