from setuptools import setup

setup(
    name="birdClassification",
    version="0.1",
    description="Focus on DFP and attempt to improve the results on several games.",
    packages=["doom", "dfp"],
    requires=["setuptools", "wheel"],
    install_requires=[
        "backcall==0.2.0",
        "black==21.11b1",
        "click==8.0.3",
        "cycler==0.11.0",
        "dataclasses==0.8",
        "decorator==5.1.0",
        "entrypoints==0.3",
        "importlib-metadata==4.8.2",
        "ipykernel==5.5.6",
        "ipython==7.16.2",
        "ipython-genutils==0.2.0",
        "jedi==0.17.2",
        "jupyter-client==7.1.0",
        "jupyter-core==4.9.1",
        "kiwisolver==1.3.1",
        "matplotlib==3.3.4",
        "mypy-extensions==0.4.3",
        "nest-asyncio==1.5.4",
        "numpy==1.19.5",
        "opencv-python==4.5.4.60",
        "parso==0.7.1",
        "pathspec==0.9.0",
        "pexpect==4.8.0",
        "pickleshare==0.7.5",
        "Pillow==8.4.0",
        "platformdirs==2.4.0",
        "prompt-toolkit==3.0.23",
        "protobuf==3.19.1",
        "ptyprocess==0.7.0",
        "Pygments==2.10.0",
        "pyparsing==3.0.6",
        "python-dateutil==2.8.2",
        "pyzmq==22.3.0",
        "regex==2021.11.10",
        "scipy==1.5.4",
        "six==1.16.0",
        "tensorflow==1.0.0",
        "tomli==1.2.2",
        "tornado==6.1",
        "traitlets==4.3.3",
        "typed-ast==1.5.1",
        "typing_extensions==4.0.1",
        "vizdoom==1.1.11",
        "wcwidth==0.2.5",
        "zipp==3.6.0",
    ],
    extras_require={
        "dev": ["tqdm", "ipykernel", "black"],
    },
    entry_points={
        "console_scripts": [
            # For Doom
            "evaluate_doom=doom.evaluate:evaluate_cli",
        ]
    },
)
