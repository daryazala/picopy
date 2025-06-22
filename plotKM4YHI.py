# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import plotObs
import readObs
import matplotlib.pyplot as plt
import pandas as pd
import glob
import hytraj

fnames = glob.glob("/home/expai/project/data/PBA*")
print(fnames)

obs = readObs.read_custom_csv(fnames[0])
print(obs)
print(obs.index)
plotObs.plot_lat_lon(obs)


tdumpnames = glob.glob("/home/expai/project/data/tdump*")
print(tdumpnames)

traj = hytraj.open_dataset(tdumpnames[0])
print(traj)
plotObs.plot_lat_lon(traj)

fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
plotObs.plot_lat_lon_mod(obs, ax, None)
plotObs.plot_lat_lon_mod(traj, ax, clr='red')
ax.set_xlim(-100,100)
ax.set_ylim(25,60)
plt.show()


#obs.time = pd.to_datetime(obs.time)
#plt.scatter(obs.time,obs.latitude)

#fig, axs = plt.subplots(figsize=(8,4))
#axs.scatter(obs.time,obs.altitude)


# Create the first figure and Axes for time vs latitude
fig1, ax1 = plt.subplots()
ax1.scatter(obs.time, obs.latitude)
ax1.set_xlabel('Time')
ax1.set_ylabel('Latitude')
ax1.set_title('Time vs Latitude')

# Create the second figure and Axes for time vs altitude
fig2, ax2 = plt.subplots()
ax2.scatter(obs.time, obs.altitude)
ax2.set_xlabel('Time')
ax2.set_ylabel('Altitude')
ax2.set_title('Time vs Altitude')

plt.show()