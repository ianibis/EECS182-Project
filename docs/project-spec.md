Calibration draws
EEGNet decoder
EEGNet regression head

Calibration draws: for each test session and k = 2, 5, 10 and 20, sample 10 sets of k stimulation on trials; the remaining on trials are the query trials. Save to draws.json
EEGNet regression decoder (braindecode) on 1 kHz LFP, predicting kinematics per 50 ms window, trained on stimulation off trials
Adapt braindecode's EEGNet from classification to regression with one kinematic output per 50 ms window
Write the training loop: early stopping on validation trials, and NaN kinematics masked out of the loss
Train 5 seeds and compare held out off R² with ridge (3.1)
Traditional pipeline: PARRM cleaning, then ridge refit on the k calibration trials of each draw

If you have a GRU that can predict your velocities already, what is the point of building/tuning the other models within your project? Talked about how the GRU is only used on the non-stimulation data as of right now. 
We’re using the GRU model to transform the other data into velocities 
Side experiment: Can you train a vanilla MLP to solve this problem for you without a transformer?
Potential suggestion: Change out the oracle for the MLP
