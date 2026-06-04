from setuptools import setup, find_packages

setup(
    name="planar-peg",
    version="0.1.0",
    description="A 2.5D physical environment for planar peg insertion using MuJoCo and Gymnasium.",
    author="DeepMind Pair Programmer",
    packages=find_packages(),
    install_requires=[
        "gymnasium-robotics",
        "numpy>=1.21.0",
        "mujoco>=3.0.0",
        "pygame>=2.1.0",  # Used for gamepad/keyboard teleoperation
        "h5py>=3.6.0",    # Used for demonstration data logging
        "opencv-python",  # Used for visualization of demonstration data
    ],
    python_requires=">=3.10",
)
