/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *
 *  Changed Jul 01, 2018 by
 *              Arthur Pichlkostner
 */

#include <math.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

constexpr bool PREDICTION_ERROR_GAUSSIAN = true;
constexpr bool ASSOCIATE_OBSERVATIONS_TO_MAPPREDICTION = false;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles_ = 20;

  default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles_; ++i) {
    double sample_x, sample_y, sample_theta;

    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    Particle new_particle{i, sample_x, sample_y, sample_theta, 1};

    particles.push_back(new_particle);
  }
  weights_.resize(num_particles_);
  is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  constexpr double MIN_YAWRATE = 0.0001;
  default_random_engine gen;

  if (PREDICTION_ERROR_GAUSSIAN) {
    for (int i = 0; i < num_particles_; ++i) {
      double theta = particles[i].theta;

      if (abs(yaw_rate) > MIN_YAWRATE) {
        particles[i].x += velocity / yaw_rate *
                          (sin(theta + yaw_rate * delta_t) - sin(theta));
        particles[i].y += velocity / yaw_rate *
                          (cos(theta) - cos(theta + yaw_rate * delta_t));
        particles[i].theta += yaw_rate * delta_t;
      } else {
        particles[i].x += velocity * cos(theta) * delta_t;
        particles[i].y += velocity * sin(theta) * delta_t;
      }

// add uncertainty to x, y and theta
#if 1
      normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
      normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
#else
      // better result with smaller stddev
      normal_distribution<double> dist_x(particles[i].x, std_pos[0] * 0.2);
      normal_distribution<double> dist_y(particles[i].y, std_pos[1] * 0.2);
#endif
      normal_distribution<double> dist_theta(particles[i].theta,
                                             std_pos[2] * 0.5);

      particles[i].x = dist_x(gen);
      particles[i].y = dist_y(gen);
      particles[i].theta = dist_theta(gen);
    }
  } else {
    // add uncertainty to v an yaw
    normal_distribution<double> dist_v(velocity, 1);
    normal_distribution<double> dist_yaw(yaw_rate, 0.1);

    for (int i = 0; i < num_particles_; ++i) {
      double theta = particles[i].theta;
      const double new_v = dist_v(gen);
      const double new_yaw = dist_yaw(gen);

      if (abs(new_yaw) > MIN_YAWRATE) {
        particles[i].x +=
            new_v / new_yaw * (sin(theta + new_yaw * delta_t) - sin(theta));
        particles[i].y +=
            new_v / new_yaw * (cos(theta) - cos(theta + new_yaw * delta_t));
        particles[i].theta += new_yaw * delta_t;
      } else {
        particles[i].x += new_v * cos(theta) * delta_t;
        particles[i].y += new_v * sin(theta) * delta_t;
      }
#if 0
      // add uncertainty to x, y and theta
      // filter don't works without additional gaussian noise on final position
      normal_distribution<double> dist_x(particles[i].x, std_pos[0] * 0.2);
      normal_distribution<double> dist_y(particles[i].y, std_pos[1] * 0.2);

      particles[i].x = dist_x(gen);
      particles[i].y = dist_y(gen);
#endif
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& ref_pos,
                                     std::vector<LandmarkObs>& to_be_mapped) {
  for (auto& rp : ref_pos) {
    double min_dist2 = std::numeric_limits<double>::max();

    // find observation with minimum distance to prediction
    for (const auto& tbm : to_be_mapped) {
      double dist2 =
          (tbm.x - rp.x) * (tbm.x - rp.x) + (tbm.y - rp.y) * (tbm.y - rp.y);

      if (min_dist2 > dist2) {
        rp.id = tbm.id;
        min_dist2 = dist2;
      }
    }
  }
}

std::vector<LandmarkObs> ParticleFilter::local2global(
    const Particle& p, const std::vector<LandmarkObs>& local_pos) {
  std::vector<LandmarkObs> global_pos;
  int cnt = 0;

  for (auto& loc : local_pos) {
    LandmarkObs lmo;
    lmo.x = p.x + cos(p.theta) * loc.x - sin(p.theta) * loc.y;
    lmo.y = p.y + sin(p.theta) * loc.x + cos(p.theta) * loc.y;
    lmo.id = cnt++;

    global_pos.push_back(lmo);
  }

  return global_pos;
}

std::vector<LandmarkObs> ParticleFilter::inRange(const Particle& p,
                                                 const Map& map_landmarks,
                                                 double sensor_range) {
  double sensor_range2 = sensor_range * sensor_range;
  std::vector<LandmarkObs> nearPos;
  int cnt = 0;

  for (const auto& lm : map_landmarks.landmark_list) {
    double dist2 =
        (lm.x_f - p.x) * (lm.x_f - p.x) + (lm.y_f - p.y) * (lm.y_f - p.y);

    if (dist2 <= sensor_range2) {
      LandmarkObs new_lm;

      new_lm.x = lm.x_f;
      new_lm.y = lm.y_f;
      new_lm.id = cnt++;

      nearPos.push_back(new_lm);
    }
  }

  return nearPos;
}

void ParticleFilter::calcNewWeights(
    Particle* p, double std_landmark[],
    const std::vector<LandmarkObs>& landmarks1,
    const std::vector<LandmarkObs>& landmarks2) {
  const double sig_x = std_landmark[0];
  const double sig_y = std_landmark[1];
  const double gauss_norm = (1. / (2. * M_PI * sig_x * sig_y));

  if ((landmarks1.size() > 0) && (landmarks2.size() > 0)) {
    p->weight = 1.;

    for (auto& pred : landmarks1) {
      double mu_x = landmarks2[pred.id].x;
      double mu_y = landmarks2[pred.id].y;

      double exponent =
          ((pred.x - mu_x) * (pred.x - mu_x)) / (2. * sig_x * sig_x) +
          ((pred.y - mu_y) * (pred.y - mu_y)) / (2. * sig_y * sig_y);
      double weight = gauss_norm * exp(-exponent);

      p->weight *= weight;
    }
  } else {
    // if there are not predicted landmarks or no measurements the weight should
    // be 0 in our case
    p->weight = 0.;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  auto weights_iterator = weights_.begin();

  for (auto& p : particles) {
    // tranform from local to global coordinate system
    std::vector<LandmarkObs> obs_glob = local2global(p, observations);
    // build vector with landmarks in range of sensor
    std::vector<LandmarkObs> predictions =
        inRange(p, map_landmarks, sensor_range);

    if (ASSOCIATE_OBSERVATIONS_TO_MAPPREDICTION) {
      // association of observations to landmarks
      dataAssociation(predictions, obs_glob);
      // calculate new weights
      calcNewWeights(&p, std_landmark, predictions, obs_glob);
    } else {
      // association of landmarks to observations
      dataAssociation(obs_glob, predictions);
      // calculate new weights
      calcNewWeights(&p, std_landmark, obs_glob, predictions);
    }

    // fill weight vector to make resample method easier
    *weights_iterator = p.weight;
    weights_iterator++;
  }
}

void ParticleFilter::resample() {
  default_random_engine gen;
  discrete_distribution<int> distr(weights_.begin(), weights_.end());

  vector<Particle> res_part;

  for (auto part : particles) {
    res_part.push_back(particles[distr(gen)]);
  }

  particles = res_part;
}

Particle ParticleFilter::SetAssociations(Particle& particle,
                                         const std::vector<int>& associations,
                                         const std::vector<double>& sense_x,
                                         const std::vector<double>& sense_y) {
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
