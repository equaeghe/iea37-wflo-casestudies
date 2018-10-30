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


def proximity_repulsion(vectors, min_dist=2.0):
    """Generates repulsion steps for turbines that are too close to each other

    min_dist is in units of rotor diameter

    """
    # Calculate matrix of pairwise turbine distances
    dists = np.sqrt(vectors.x ** 2 + vectors.y ** 2)

    # Generate pairwise repulsion steps
    step_size = (min_dist - dists) / 2
        # signed; will be so used if negative; should be just enough to fix
        # the issue (unless there is opposing repulsion as well)
    repulsion_steps = np.recarray(dists.shape, wflocs.xy_pair)
    repulsion_steps.x = np.where(step_size > 0, step_size * vectors.x, 0)
    repulsion_steps.y = np.where(step_size > 0, step_size * vectors.y, 0)

    # Combine repulsion steps in all directions for each turbine
    repulsion_leap = np.recarray(vectors.shape[0], wflocs.xy_pair)
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
    layout = np.recarray((number_of_turbines,), wflocs.xy_pair)
    magnitudes = farm_radius * np.random.rand(number_of_turbines)
    angles = 2 * np.pi * np.random.rand(number_of_turbines)
        # TODO: sample according to wind resource and rotate 90°
    layout.x = magnitudes * np.cos(angles)
    layout.y = magnitudes * np.sin(angles)

    return constraint_fixer(layout, farm_radius)


def randomize_steps(step, threshold=.5):
    """Randomly reorient some of the steps to increase domain search"""
    restep = np.copy(step).view(np.recarray)
    rnd = 2 * np.random.rand(len(step)) - 1
    restep.x = np.where(np.abs(rnd) < threshold, step.x, step.y * np.sign(rnd))
    restep.y = np.where(np.abs(rnd) < threshold, step.y, -step.x * np.sign(rnd))

    return restep


def search_equilibrium(layout, farm_radius: float, iterations: int,
                       downwind_vectors, wind_freq,
                       wind_speed, turb_ci, turb_co):
    """Iteratively move towards a (hoped) equilibrium layout

    The initial layout is iteratively modified, keeping it inside the site,
    until the gradients that define how much the turbines move fall below the
    tolerance.

    Return the final layout.

    """
    # useful parameters
    n = len(layout)
    scale_multiplier = .89 # TODO: make this configurable, it has an impact
                           # (optimal value depends on size as well)
                           # too high and everything lands on the border
                           # val ≠ 1 seem incompatible with randomized steps
    step_scaler = 56 * farm_radius / n
    down_coeff, cross_coeff, back_coeff = 0, 0, 0
    m = 4
    # BEST non-interleaved
    # for 16 turbine case: 0, 1, 0 -> 14.1%
    # for 36 turbine case: .45, 0, .55
    # for 64 turbine case: .5, 0, .5; 0, 1, 0
    # iteration & quality tracking variables
    # BEST interleaved
    # for 64: step_scaler =  56 * farm_radius / n, m = 4, scale_multiplier = .89 -> 22.5%
    # for 36: step_scaler =  52 * farm_radius / n, m = 2, scale_multiplier = .9 -> 21.6%
    # for 16: step_scaler =  16 * farm_radius / n, m = 3, scale_multiplier = .95 -> 14.3%
    cur_opt = 1.0
    best_step = -1
    best_opt = 1.0
    best_step = -1
    max_step = np.nan
    max_repulsion = np.nan
    steps, repulsions, retrenchments = 0, 0, 0
    # layouts we track
    new = np.copy(layout).view(np.recarray)
    best = np.copy(layout).view(np.recarray)
    while steps + retrenchments < iterations:
        if steps % m is 0:
            down_coeff, cross_coeff, back_coeff = (.5, 0, .5) if m > 0 else (0, 1, 0)
        else:
            down_coeff, cross_coeff, back_coeff = (0, 1, 0) if m > 0 else (.5, 0, .5)
        new = np.copy(layout).view(np.recarray)
        # evaluate current layout
        vectors = wflocs.turbine_vectors(layout)
        repulsion = proximity_repulsion(vectors)
        max_repulsion = np.max(np.sqrt(repulsion.x ** 2 + repulsion.y ** 2))
        if max_repulsion > 0:  # distance constraint violated!
            # move so that locally repulsion becomes zero
            # (new repulsion is possible)
#            print(steps, repulsions, retrenchments,
#                  'REPULSIVE layout', max_repulsion)
            new.x += repulsion.x
            new.y += repulsion.y
            new = constraint_fixer(new, farm_radius)
            layout = np.copy(new).view(np.recarray)
            max_step, cur_opt = np.nan, np.nan
            repulsions += 1
            continue
        # non-repulsive layout
        frame_vectors = wflocs.wind_frames(vectors, downwind_vectors)
        loss, blame_fractions = wflocs.gaussian_wake(frame_vectors)
        powers = wflocs.power(loss, wind_speed, turb_ci, turb_co)
        cur_opt = 1 - np.sum(powers @ wind_freq) / n
        print(steps, repulsions, sep=',', end=' ')
        if cur_opt > 1.2 * best_opt:  # stopping crit
            print(steps, repulsions, retrenchments, 'HOPELESS degradation')
            break
        if cur_opt > 1.07 * best_opt:
#            step_scaler /= scale_multiplier
            print(steps, repulsions, retrenchments,
                  'WORRYING degradation', '| unit step now', step_scaler)
            layout = np.copy(best).view(np.recarray)
            cur_opt = best_opt
            max_step = np.nan
            retrenchments += 1
            continue
        if cur_opt < best_opt:  # new best layout
            step_scaler *= scale_multiplier
            best = np.copy(layout).view(np.recarray)
            best_step = steps
            best_opt = cur_opt
            best_step = steps
            print(steps, repulsions, retrenchments,
                  '*** BEST layout', '| unit step now', step_scaler)
#        step_scaler *= scale_multiplier
        # update the layout according to pseudo-gradients
        wakeless_pwr = wflocs.wakeless_pwr(wind_speed, turb_ci, turb_co)
        step = wflocs.pseudo_gradient(wind_freq,
                                      downwind_vectors, vectors, frame_vectors,
                                      wakeless_pwr - powers, blame_fractions,
                                      down_coeff, cross_coeff, back_coeff)
#        step = randomize_steps(step, .75)
        # remove common drift
        step.x -= np.mean(step.x)
        step.y -= np.mean(step.y)
        # calculate new layout
        new.x += step_scaler * step.x
        new.y += step_scaler * step.y
        new = constraint_fixer(new, farm_radius)
        max_step = np.max(np.sqrt((new.x - layout.x) ** 2 +
                                  (new.y - layout.y) ** 2))
        # update loop variables
        layout = np.copy(new).view(np.recarray)
        print(steps, repulsions, retrenchments, 'STEP taken', max_step)
        steps += 1

    print(best_step, best_opt)
    print('\n*', best_step, best_opt)
    return best
