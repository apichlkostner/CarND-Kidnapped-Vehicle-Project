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

    //cout << new_particle.id << " x = " << new_particle.x << " y = " << new_particle.y << " theta = " << new_particle.theta << endl;

    particles.push_back(new_particle);
  }
  weights_.resize(num_particles_);
  is_initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  default_random_engine gen;

  for (int i = 0; i < num_particles_; ++i) {
    double theta = particles[i].theta;

    if (abs(yaw_rate) > 0.0001) {
      particles[i].x +=
          velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      particles[i].y +=
          velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      particles[i].x += velocity * cos(theta) * delta_t;
      particles[i].y += velocity * sin(theta) * delta_t;
    }

    normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

    const double new_x = dist_x(gen);
    const double new_y = dist_y(gen);
    const double new_theta = dist_theta(gen);

    particles[i].x = new_x;
    particles[i].y = new_y;
    particles[i].theta = new_theta;
  }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& predictions,
                                     std::vector<LandmarkObs>& observations) {
  for (auto& pred : predictions) {
    float min_dist2 = 1e20;

    for (auto& obs : observations) {
      float dist2 = (obs.x - pred.x) * (obs.x - pred.x) +
                    (obs.y - pred.y) * (obs.y - pred.y);

      if (min_dist2 > dist2) {
        pred.id = obs.id;
        min_dist2 = dist2;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  float sensor_range2 = sensor_range * sensor_range;
  int particle_nr = 0;

  for (auto& p : particles) {
    std::vector<LandmarkObs> obs_glob;

    int i = 0;
    for (auto& obs : observations) {
      LandmarkObs lmo;
      lmo.x = p.x + cos(p.theta) * obs.x - sin(p.theta) * obs.y;
      lmo.y = p.y + sin(p.theta) * obs.x + cos(p.theta) * obs.y;
      lmo.id = i++;

      obs_glob.push_back(lmo);
    }

    std::vector<LandmarkObs> predictions;
    int cnt = 0;
    for (auto& lm : map_landmarks.landmark_list) {
      float dist2 =
          (lm.x_f - p.x) * (lm.x_f - p.x) + (lm.y_f - p.y) * (lm.y_f - p.y);
      if (dist2 <= sensor_range2) {
        LandmarkObs new_lm;

        new_lm.x = lm.x_f;
        new_lm.y = lm.y_f;
        new_lm.id = cnt++;

        predictions.push_back(new_lm);
      }
    }

    dataAssociation(predictions, obs_glob);

    const double sig_x = std_landmark[0];
    const double sig_y = std_landmark[1];
    double gauss_norm = (1. / (2. * M_PI * sig_x * sig_y));

    if (predictions.size() > 0)
      p.weight = 1.;
    else
      p.weight = 0.;

    for (auto& pred : predictions) {
      double mu_x = obs_glob[pred.id].x;
      double mu_y = obs_glob[pred.id].y;
     
      double exponent =
          ((pred.x - mu_x) * (pred.x - mu_x)) / (2. * sig_x * sig_x) +
          ((pred.y - mu_y) * (pred.y - mu_y)) / (2. * sig_y * sig_y);
      double weight = gauss_norm * exp(-exponent);

      p.weight *= weight;
    }

    weights_[particle_nr++] = p.weight;
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
