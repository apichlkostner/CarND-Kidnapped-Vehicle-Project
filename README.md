# Particle filter - kidnapped vehicle project

# Summary

This is a project for an online course for selfdriving cars.

The steps of this project are the following:

- Implementation of a particle filter to find the correct car position from measurements using a map
    - Initialization of particles with the position from a GPS sensor    
    - Predict new positions of the particles using a system model
    - Predict sensor measurements from predicted position and map
    - Add random noise to the new state
    - Read sensor measurements
    - Associate measurements with predictions
    - Calculate the likelihood of the sensor measurements
    - Weight the particles proportional to their likelihood
    - Resample the particles
- Run the particle filter connected to a simulator
- Evaluate the performance of the filter

# Results

The result with 200 particles and smaller process standard deviation than in method call from `main()`:
![](docu/result.png)

While running the filter:
![](docu/simulation.png)

Error and processing time with different number of particles:

|nr of particle | err x | err y | err yaw | time |
|---------------|-------|-------|---------|------|
| 20            | 0.131 | 0.127 | 0.005   | 49s  |
| 200           | 0.112 | 0.107 | 0.004   | 49s  |
| 500           | 0.109 | 0.102 | 0.004   | 49s  |
| 1000          | 0.108 | 0.100 | 0.004   | 49s  |
| 1500          | 0.110 | 0.101 | 0.004   | 51s  |
| 2000          | 0.107 | 0.100 | 0.004   | 72s  |
| 2500          | 0.107 | 0.100 | 0.004   | 92s  |
| 3000          | 0.110 | 0.102 | 0.004   | 98s  |
| 4000          | 0.110 | 0.102 | 0.004   | 109s |

The 49s are the base time for running the simulator and communication. With less than 1500 particles the calculation of the filter is faster than the pause times of the simulator.

# Experiments

## Add noise to input v and psi_dot

In the proposed version the error model adds gaussian noise to the system state after the calculating the system equations.
It was tried to add gaussian noise to the input variable v ans psi_dot instead. This can be switched in the code with `PREDICTION_ERROR_GAUSSIAN=false`.

With this modified error model the filter don't works when the car drives strong curves. It seems that it is necessary to change the x and y coordinates independently during the prediction step.

## Changing the standard deviation of the system error model

In the base code the process noise is the same as the sensor noise of the GPS which is used to initialize the filter. Using smaller standard deviations gives smaller mse:

200 particle 0.087, 0.083, 0.003

## Use measurements or predicted measurements as base values

It's possible to use the measurements as base, associate a landmark to each of them, and then calculate the likelihood for every measurement. Or the predicted landmarks which should be found by the sensors could be used as base and a measurement could be mapped to each landmark.

The method can be changed in the code with `ASSOCIATE_BASE_ARE_OBSERVATIONS`.

In our case there is no difference in the result since the initial position is already known relatively good and the measurements always find the landmarks with only some noise.

# Running the Code
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and install uWebSocketIO for either Linux or Mac systems. For windows you can use either Docker, VMware, or even Windows 10 Bash on Ubuntu to install uWebSocketIO.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./particle_filter

Alternatively some scripts have been included to streamline this process, these can be leveraged by executing the following in the top directory of the project:

1. ./clean.sh
2. ./build.sh
3. ./run.sh

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

Note that the programs that need to be written to accomplish the project are src/particle_filter.cpp, and particle_filter.h

The program main.cpp has already been filled out, but feel free to modify it.

Here is the main protocol that main.cpp uses for uWebSocketIO in communicating with the simulator.

INPUT: values provided by the simulator to the c++ program

// sense noisy position data from the simulator

["sense_x"]

["sense_y"]

["sense_theta"]

// get the previous velocity and yaw rate to predict the particle's transitioned state

["previous_velocity"]

["previous_yawrate"]

// receive noisy observation data from the simulator, in a respective list of x/y values

["sense_observations_x"]

["sense_observations_y"]


OUTPUT: values provided by the c++ program to the simulator

// best particle values used for calculating the error evaluation

["best_particle_x"]

["best_particle_y"]

["best_particle_theta"]

//Optional message data used for debugging particle's sensing and associations

// for respective (x,y) sensed positions ID label

["best_particle_associations"]

// for respective (x,y) sensed positions

["best_particle_sense_x"] <= list of sensed x positions

["best_particle_sense_y"] <= list of sensed y positions


Your job is to build out the methods in `particle_filter.cpp` until the simulator output says:

```
Success! Your particle filter passed!
```

# Implementing the Particle Filter
The directory structure of this repository is as follows:

```
root
|   build.sh
|   clean.sh
|   CMakeLists.txt
|   README.md
|   run.sh
|
|___data
|   |   
|   |   map_data.txt
|   
|   
|___src
    |   helper_functions.h
    |   main.cpp
    |   map.h
    |   particle_filter.cpp
    |   particle_filter.h
```

The only file you should modify is `particle_filter.cpp` in the `src` directory. The file contains the scaffolding of a `ParticleFilter` class and some associated methods. Read through the code, the comments, and the header file `particle_filter.h` to get a sense for what this code is expected to do.

If you are interested, take a look at `src/main.cpp` as well. This file contains the code that will actually be running your particle filter and calling the associated methods.

## Inputs to the Particle Filter
You can find the inputs to the particle filter in the `data` directory.

#### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

### All other data the simulator provides, such as observations and controls.

> * Map data provided by 3D Mapping Solutions GmbH.