from gymnasium.envs.registration import register

# Register the custom Gymnasium environment
register(
    id="PlanarPegInsertion-v0",
    entry_point="planar_peg.envs.peg_env:PlanarPegEnv",
    max_episode_steps=300,
)
