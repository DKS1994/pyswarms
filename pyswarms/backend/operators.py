# -*- coding: utf-8 -*-

"""
Swarm Operation Backend

This module abstracts various operations in the swarm such as updating
the personal best, finding neighbors, etc. You can use these methods
to specify how the swarm will behave.
"""

# Import from stdlib
import logging

# Import modules
import numpy as np
from scipy.spatial import cKDTree

# Create a logger
logger = logging.getLogger(__name__)

def update_pbest(swarm):
    """Takes a swarm instance and updates the personal best scores
    
    You can use this method to update your personal best positions.

    ..code-block :: python

        import pyswarms.backend as P
        from pyswarms.backend.swarms import Swarm

        my_swarm = P.create_swarm(n_particles, dimensions)

        # Inside the for-loop...
        for i in range(iters):
            # It updates the swarm internally
            my_swarm.pbest_pos, my_swarm.pbest_cost = P.update_pbest(my_swarm)

    It updates your :code:`current_pbest` with the personal bests acquired by
    comparing the (1) cost of the current positions and the (2) personal
    bests your swarm has attained.
    
    If the cost of the current position is less than the cost of the personal
    best, then the current position replaces the previous personal best
    position.

    Parameters
    ----------
    swarm : pyswarms.backend.swarm.Swarm
        a Swarm instance

    Returns
    -------
    numpy.ndarray
        New personal best positions of shape :code:`(n_particles, n_dimensions)`
    numpy.ndarray
        New personal best costs of shape :code:`(n_particles,)`
    """
    try:
        # Infer dimensions from positions
        dimensions = swarm.dimensions
        # Create a 1-D and 2-D mask based from comparisons
        mask_cost = (swarm.current_cost < swarm.pbest_cost)
        mask_pos = np.repeat(mask_cost[:, np.newaxis], swarm.dimensions, axis=1)
        # Apply masks
        new_pbest_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
        new_pbest_cost = np.where(~mask_cost, swarm.pbest_cost, swarm.current_cost)
    except AttributeError:
        msg = 'Please pass a Swarm class. You passed {}'.format(type(swarm))
        logger.error(msg)
        raise
    else:
        return (new_pbest_pos, new_pbest_cost)

def update_gbest_neighborhood(swarm, p, k):
    """Updates the global best using a neighborhood approach

    This uses the cKDTree method from :code:`scipy` to obtain the nearest
    neighbours

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    k : int
        number of neighbors to be considered. Must be a
        positive integer less than :code:`n_particles`
    p: int {1,2}
        the Minkowski p-norm to use. 1 is the
        sum-of-absolute values (or L1 distance) while 2 is
        the Euclidean (or L2) distance.

    Returns
    -------
    numpy.ndarray
        Best position of shape :code:`(n_dimensions, )`
    float
        Best cost
    """
    try:
        # Obtain the nearest-neighbors for each particle
        tree = cKDTree(swarm.position)
        _, idx = tree.query(swarm.position, p=p, k=k)

        # Map the computed costs to the neighbour indices and take the
        # argmin. If k-neighbors is equal to 1, then the swarm acts
        # independently of each other.
        if k == 1:
            # The minimum index is itself, no mapping needed.
            best_neighbor = swarm.pbest_cost[idx][:, np.newaxis].argmin(axis=1)
        else:
            idx_min = swarm.pbest_cost[idx].argmin(axis=1)
            best_neighbor = idx[np.arange(len(idx)), idx_min]
        # Obtain best cost and position
        best_cost = np.min(swarm.pbest_cost[best_neighbor])
        best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost[best_neighbor])]
    except AttributeError:
        msg = 'Please pass a Swarm class. You passed {}'.format(type(swarm))
        logger.error(msg)
        raise
    else:
        return (best_pos, best_cost)
