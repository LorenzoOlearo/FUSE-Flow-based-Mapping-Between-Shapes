# FlowMatching4Matching

In this repo, you can find a clean implementation of Geometry Distributions, and a new implementation of a similar representation based on the FlowMatching approach.
In general, we propose a representation that leverages any feature as input, in particular distances between landmarks.

to run and create a model, run
python main.py --config configs/model/config.json

In the config, you can modify the features that you want to use and the model. FM will produce Flow matching trajectories, while Geomdist will use EDM trajectories. This code will produce a folder in which we store:
- The checkpoints of the model that warp the Gaussian into features.
- The features onto which the learning has been computed.
- The features generated at inference
- The log file


To run the main on a dataset, you can run 
python scripts/run_dataset.py --config configs/matching/faust_ldmk.json

After training on the dataset, test the performance on matching, you can run
python scripts/matching_scripts.py --config configs/matching/faust_ldmk.json

