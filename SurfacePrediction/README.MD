
### Robot Surface Prediction ###

Implemented as an entry to [this](https://www.kaggle.com/c/career-con-2019/overview) Kaggle competition. My entry ended up in the 3rd place on the leaderboard

Problem: Use sensor measurements (accel, gyro) and predict the surface on which a robot is moving on.

Features:
Used the Accel, Gyro features as it is. Using raw orientation features did not make sense to me, so I used the first order difference of the orientation features (after converting them to Euler angles), so that any vibrations caused by various surfaces are captured. 
Also used FFT of Accel/Gyro measurements as features.

CV:
As discussed by multiple people on the forums, GroupKFold makes sense here. k=3 gave me ~0.65-0.7 CV scores which translated to ~0.8 on the Public leaderboard.

Model:
1D Convolution based NN. I saw slightly better results with the SeparableConv1D implementation in Keras, but I'm sure the simpler Conv1D layer if tuned properly should perform equally well.
