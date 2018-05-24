# -*- coding: utf-8 -*-

"""
A Star Network Topology

This class implements a star topology where all particles are connected to
one another. This social behavior is often found in GlobalBest PSO
optimizers.
"""

# Import from stdlib
import logging

# Import modules
import numpy as np

# Import from package
from .base import Topology

# Create a logger
logger = logging.getLogger(__name__)

class Star(Topology):

    def __init__(self):
        super(Star, self).__init__()

    def compute_best_particle(self, swarm):
        """Obtains the global best cost and position based on a star topology

        This method takes the current pbest_pos and pbest_cost, then returns
        the minimum cost and position from the matrix. It should be used in
        tandem with an if statement

        .. code-block:: python

            import pyswarms.backend as P
            from pyswarms.backend.swarms import Swarm
            from pyswarm.backend.topology import Star

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Star()

            # If the minima of the pbest_cost is less than the best_cost
            if np.min(pbest_cost) < best_cost:
                # Update best_cost and position
                swarm.best_pos, swarm.best_cost = my_topology.compute_best_particle(my_swarm)

        Parameters
        ----------
        swarm : pyswarms.backend.swarm.Swarm
            a Swarm instance

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        try:
            best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
            best_cost = np.min(swarm.pbest_cost)
        except AttributeError:
            msg = 'Please pass a Swarm class. You passed {}'.format(type(swarm))
            logger.error(msg)
            raise
        else:
            return (best_pos, best_cost)

    def update_velocity(self, swarm, clamp):
        """Updates the velocity matrix

        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.
        
        A sample usage can be seen with the following:

        .. code-block :: python

            import pyswarms.backend as P
            from pyswarms.swarms.backend import Swarm
            from pyswarms.backend.topology import Star

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Star()

            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = my_topology.update_velocity(my_swarm, clamp)

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        clamp : tuple of floats (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.

        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        try:
            # Prepare parameters
            swarm_size = swarm.position.shape
            c1 = swarm.behavior['c1']
            c2 = swarm.behavior['c2']
            w = swarm.behavior['w']
            # Compute for cognitive and social terms
            cognitive = (c1 * np.random.uniform(0,1, swarm_size) * (swarm.pbest_pos - swarm.position))
            social = (c2 * np.random.uniform(0, 1, swarm_size) * (swarm.best_pos - swarm.position))
            # Compute temp velocity (subject to clamping if possible)
            temp_velocity = (w * swarm.velocity) + cognitive + social

            if clamp is None:
                updated_velocity = temp_velocity
            else:
                min_velocity, max_velocity = clamp
                mask = np.logical_and(temp_velocity >= min_velocity,
                                    temp_velocity <= max_velocity)
                updated_velocity = np.where(~mask, swarm.velocity, temp_velocity)
        except AttributeError:
            msg = 'Please pass a Swarm class. You passed {}'.format(type(swarm))
            logger.error(msg)
            raise
        except KeyError:
            msg = 'Missing keyword in swarm.behavior'
            logger.error(msg)
            raise
        else:
            return updated_velocity

    def update_position(self, swarm, bounds):
        """Updates the position matrix

        This method updates the position matrix given the current position and
        the velocity. If bounded, it waives updating the position.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.

        Returns
        -------
        numpy.ndarray
            New position-matrix
        """
        try:
            temp_position = swarm.position.copy()
            temp_position += swarm.velocity

            if bounds is not None:
                lb, ub = bounds
                min_bounds = np.repeat(np.array(lb)[np.newaxis, :], swarm.n_particles, axis=0)
                max_bounds = np.repeat(np.array(ub)[np.newaxis, :], swarm.n_particles, axis=0)
                mask = (np.all(min_bounds <= temp_position, axis=1)
                    * np.all(temp_position <= max_bounds, axis=1))
                mask = np.repeat(mask[:, np.newaxis], swarm.dimensions, axis=1)
                temp_position = np.where(~mask, swarm.position, temp_position)
            position = temp_position
        except AttributeError:
            msg = 'Please pass a Swarm class. You passed {}'.format(type(swarm))
            logger.error(msg)
            raise
        else:
            return position