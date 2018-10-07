import numpy as np
import IEA37_wflocs_basis as wflocs


def constraint_fixer(layout, farm_radius: float):
    """Fixes the coordinate pairs that fall outside the site

    The coordinate_pairs must be a recarray of wflocs.coordinate.
    The coordinates that fall outside are scaled to fall on the border.

    """
    magnitudes = np.sqrt(layout.x ** 2 + layout.y ** 2)
    outside = magnitudes > farm_radius
    scaling = farm_radius / magnitudes
    layout.x = np.where(outside, layout.x * scaling, layout.x)
    layout.y = np.where(outside, layout.y * scaling, layout.y)

    return layout


def proximity_repulsion(layout, min_dist=2.0):
    """Generates repulsion steps for turbines that are too close to each other

    min_dist is in units of rotor diameter

    """
    # Calculate matrix of vectors between all pairs of turbines
    position_matrix = np.tile(layout, (len(layout), 1))
    vectors = np.recarray(position_matrix.shape, wflocs.coordinate)
    vectors.x = position_matrix.x - position_matrix.x.T
    vectors.y = position_matrix.y - position_matrix.y.T

    # Calculate matrix of pairwise turbine distances
    dists = np.sqrt(vectors.x ** 2 + vectors.y ** 2)

    # Generate pairwise repulsion steps
    step_size = (min_dist - dists) / 2
        # signed; will be so used if negative; should be just enough to fix
        # the issue (unless there is opposing repulsion as well)
    repulsion_steps = np.recarray(dists.shape, wflocs.coordinate)
    repulsion_steps.x = np.where(step_size > 0, step_size * vectors.x, 0)
    repulsion_steps.y = np.where(step_size > 0, step_size * vectors.y, 0)

    # Combine repulsion steps in all directions for each turbine
    repulsion_leap = np.recarray(layout.shape, wflocs.coordinate)
    repulsion_leap.x = np.sum(repulsion_steps.x, 0)
    repulsion_leap.y = np.sum(repulsion_steps.y, 0)

    return repulsion_leap


def deterministic_layout(number_of_turbines: int, farm_radius: float,
                         downwind_vectors, wind_freq,):
    """Generate a layout orthogonal to average wind direction

    The idea is to put as many turbines as possible on a line through
    the origin along the direction orthogonal to the average wind direction.
    (The idea being that this will be very effective for ‘pointed’ windroses.)

    This is not yet implemented.

    """
    pass


def random_layout(number_of_turbines: int, farm_radius: float):
    """Generate a random layout within a disc-shaped site

    The farm radius should be in rotor diameter units.
    The output layout coordinates are also in those units.
    (So it is assumed that the rotor diameter is constant.)
    The coordinates are generated as an independent bivariate centered normal
    distribution with standard deviation a third the farm radius.

    """
    layout = np.recarray((number_of_turbines,), wflocs.coordinate)
    magnitudes = farm_radius * np.random.rand(number_of_turbines)
    angles = 2 * np.pi * np.random.rand(number_of_turbines)
        # TODO: sample according to wind resource and rotate 90°
    layout.x = magnitudes * np.cos(angles)
    layout.y = magnitudes * np.sin(angles)

    return constraint_fixer(layout, farm_radius)


def search_equilibrium(layout, farm_radius: float, tol: float,
                       downwind_vectors, wind_freq,
                       wind_speed, turb_ci, turb_co):
    """Iteratively move towards a (hoped) equilibrium layout

    The initial layout is iteratively modified, keeping it inside the site,
    until the gradients that define how much the turbines move fall below the
    tolerance.

    Return the final layout.

    """
    cur_opt = 1
    best_opt = 1
    n = len(layout)
    step_scaler = 2 * farm_radius / n  # TODO: try smaller and larger steps
    max_step = np.nan
    max_repulsion = np.nan
    i = 0
    new = np.copy(layout).view(np.recarray)
    best = np.copy(layout).view(np.recarray)
    while True:
        powers = wflocs.rose_power(layout, downwind_vectors, wind_speed,
                                   turb_ci, turb_co)
        cur_opt = 1 - np.sum(wind_freq * np.sum(powers, axis=1)) / n
        print(i, cur_opt, max_step, max_repulsion)
        if max_repulsion == 0.0 and cur_opt > 1.1 * best_opt:
            break
        if max_repulsion == 0.0 and cur_opt < best_opt:
            best = np.copy(layout).view(np.recarray)
        best_opt = np.minimum(cur_opt, best_opt)
        repulsion = proximity_repulsion(layout)
        repulsion_size = np.sqrt(repulsion.x ** 2 + repulsion.y ** 2)
        max_repulsion = np.max(repulsion_size)
        deficits = wflocs.rose_deficits(wind_speed, turb_ci, turb_co, powers)
        step = wflocs.pseudo_gradient(wind_freq, downwind_vectors, deficits)
        # remove common drift
        step.x -= np.mean(step.x)
        step.y -= np.mean(step.y)
        # calculate new layout
        new.x += np.where(max_repulsion > 0, repulsion.x, step_scaler * step.x)
        new.y += np.where(max_repulsion > 0, repulsion.y, step_scaler * step.y)
        new = constraint_fixer(new, farm_radius)
        # update loop variables
        step_size = np.sqrt((new.x - layout.x) ** 2 + (new.y - layout.y) ** 2)
        max_step = np.max(step_size)
        layout = np.copy(new).view(np.recarray)
        i += 1

    return best
