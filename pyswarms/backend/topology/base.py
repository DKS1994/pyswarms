# -*- coding: utf-8 -*-

"""
Base class for Topologies
"""

class Topology(object):

    def __init__(self, **kwargs):
        """Initializes the class"""
        pass

    def compute_gbest(self, swarm):
        """Computes the best particle of the swarm and returns the cost and
        position"""
        raise NotImplementedError("Topology::compute_best_particle()")

    def update_position(self, swarm):
        """Updates the swarm's position-matrix"""
        raise NotImplementedError("Topology::update_position()")

    def update_velocity(self, swarm):
        """Updates the swarm's velocity-matrix"""
        raise NotImplementedError("Topology::update_velocity()")