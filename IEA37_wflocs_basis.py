"""IEA Task 37 Combined Case Study AEP Calculation Code

Written by Nicholas F. Baker, PJ Stanley, and Jared Thomas (BYU FLOW lab)
Created 10 June 2018
Updated 11 Jul 2018 to include read-in of .yaml turb locs and wind freq dist.
Completed 26 Jul 2018 for commenting and release
Modified 22 Aug 2018 implementing multiple suggestions from Erik Quaeghebeur:
    - PEP 8 adherence for blank lines, length(<80 char), var names, docstring.
    - Altered multiple comments for clarity.
    - Used print_function for compatibility with Python 3.
    - Used structured datatype (coordinate) and recarray to couple x,y coords.
    - Removed unused variable 'sWindRose' (getTurbLocYAML).
    - Removed unecessary "if ... < 0" case (WindFrame).
    - Simplified calculations for sin/cos_wind_dir (WindFrame).
    - Eliminated unecessary calculation of 0 values (GaussianWake, DirPower).
    - Turbine diameter now drawn from <.yaml> (GaussianWake)
    - Used yaml.safe_load.
    - Modified .yaml reading syntax for brevity.
    - Removed some (now) unused array initializations.
Modified 27 Aug 2018 by Erik Quaeghebeur:
    - Cosmetic changes (whitespace).
    - Make model adimensional.
    - Vectorized for loops for significant speed increase of calcAEP.
"""

from __future__ import print_function   # For Python 3 compatibility
import numpy as np
import sys
import yaml                             # For reading .yaml files

# Structured datatypes for holding (coordinate) pairs
xy_pair = np.dtype([('x', 'f8'), ('y', 'f8')])
dc_pair = np.dtype([('d', 'f8'), ('c', 'f8')])  # downwind/crosswind


def turbine_vectors(turb_coords):
    """Calculate matrix of vectors between all pairs of turbines

    * The first array index fixes the target,
      selecting an array of outgoing vectors.
    * The second array index fixes the source,
      selecting an array of incoming vectors.

    """
    position_matrix = np.tile(turb_coords, (len(turb_coords), 1))
    # all rows of position_matrix (fixed first index) are the same
    vectors = np.recarray(position_matrix.shape, xy_pair)
    vectors.x = position_matrix.x - position_matrix.x.T
    vectors.y = position_matrix.y - position_matrix.y.T

    return vectors  # vectors[outgoing, incoming]


def downwind_vector(windrose_deg):
    """Calculate the unit vector in the downwind direction"""

    # Convert inflow wind direction
    # - from windrose (N=0, CW) to standard (E=0, CCW): 90 - wind_dir
    # - from upwind to downwind: +180
    # - from degrees to radians
    standard_rad = np.radians(90 - windrose_deg + 180)

    downwind = np.recarray(standard_rad.shape, xy_pair)
    downwind.x = np.cos(standard_rad)
    downwind.y = np.sin(standard_rad)
    return downwind  # downwind[wind directions]


def wind_frames(coords, downwinds):
    """Convert map coordinates to downwind/crosswind coordinates

    The last array index fixes the downwind direction.

    """
    frame_coords = np.recarray(coords.shape + downwinds.shape, dc_pair)
    coords = np.expand_dims(coords, -1).view(np.recarray)
    frame_coords.d = coords.x * +downwinds.x + coords.y * downwinds.y
    frame_coords.c = coords.x * -downwinds.y + coords.y * downwinds.x
    coords = np.squeeze(coords, -1).view(np.recarray)

    return frame_coords
    # frame_coords[outgoing, incoming, wind directions]


def gaussian_wake(frame_vectors):
    """Return each turbine's total speed deficit due to turbine wakes"""
    # Equations and values explained in <iea37-wakemodel.pdf>

    # Constant thrust coefficient
    CT = 4.0*1./3.*(1.0-1./3.)
    # Constant, relating to a turbulence intensity of 0.075
    k = 0.0324555

    # If the turbine of interest is downwind of the turbine generating the
    # wake, there is a wake loss; calculate it using the Simplified Bastankhah
    # Gaussian wake model
    downwind = frame_vectors.d > 0
    sigma = k*frame_vectors.d[downwind] + 1./np.sqrt(8.)
    exponent = -0.5 * (frame_vectors.c[downwind]/sigma)**2
    radical = 1. - CT/(8. * sigma**2)
    losses = np.zeros(frame_vectors.shape)
    losses[downwind] = (1.-np.sqrt(radical)) * np.exp(exponent)

    sq_losses = losses ** 2
    sq_loss = np.sum(sq_losses, axis=0)  # sum over all outgoing
    sq_loss_tiled = np.tile(np.expand_dims(sq_loss, 0),
                            (sq_losses.shape[1], 1, 1))
    b = sq_loss_tiled > 0
    blame_fractions = np.zeros(sq_losses.shape)
    blame_fractions[b] = sq_losses[b] / sq_loss_tiled[b]
    # Array holding the wake speed deficit seen at each turbine
    loss = np.sqrt(sq_loss)

    return loss, blame_fractions
    # loss[incoming, wind directions]
    # blame_fractions[outgoing, incoming, wind directions]


def ra_gaussian_wake(frame_vectors, turb_height):
    """Return each turbine's total speed deficit due to turbine wakes

    turb_height in units of turb_diam

    """
    # Equations and values explained in <iea37-wakemodel.pdf>

    # Constant thrust coefficient
    CT = 4.0*1./3.*(1.0-1./3.)
    # Constant, relating to a turbulence intensity of 0.075
    k = 0.0324555

    # If the turbine of interest is downwind of the turbine generating the
    # wake, there is a wake loss; calculate it using the Simplified Bastankhah
    # Gaussian wake model
    # 1. only downwind-coordinate-dependent
    downwind = frame_vectors.d > 0
    sigma = k*frame_vectors.d[downwind] + 1./np.sqrt(8.)
    radical = 1. - CT/(8. * sigma**2)
    # 2. also crosswind-coordinate-dependent
    c = np.tile(frame_vectors.c[downwind], (5, 1))
    # c[0] is disc center (already correct), c[1] is ‘left’, c[2] is ‘right’,
    # c[3] is ‘top’/bottom, and c[4] is mirror center
    # in units of turb_diam, turb_radius is 0.5 and turb_radius**2 is 0.25
    c[1] -= 0.5
    c[2] += 0.5
    c[3] = np.sqrt(c[0]**2 + 0.25)
    c[4] = np.sqrt(c[0]**2 + 4 * turb_height**2)
    # calculate exponential factor at all the sampled location
    e = np.exp(-0.5 * (c/sigma)**2)
    # calculate losses as rotor-averaged wake (4-point quadrature)
    # and mirror turbine for ground effect (only at rotor center)
    losses = np.zeros(frame_vectors.shape)
    λ = 2/3/np.pi
    losses[downwind] = (1.-np.sqrt(radical)) * (e[4] +
                                    λ*e[0] + (1-λ)*(e[1]/4 + e[2]/4 + e[3]/2))
    sq_losses = losses ** 2
    sq_loss = np.sum(sq_losses, axis=0)  # sum over all outgoing
    sq_loss_tiled = np.tile(np.expand_dims(sq_loss, 0),
                            (sq_losses.shape[1], 1, 1))
    b = sq_loss_tiled > 0
    blame_fractions = np.zeros(sq_losses.shape)
    blame_fractions[b] = sq_losses[b] / sq_loss_tiled[b]
    # Array holding the wake speed deficit seen at each turbine
    loss = np.sqrt(sq_loss)

    return loss, blame_fractions
    # loss[incoming, wind directions]
    # blame_fractions[outgoing, incoming, wind directions]


def power(wake_deficit, wind_speed, turb_ci, turb_co):
    """Return the power generated by each turbine"""

    # Effective windspeed is freestream multiplied by wake deficits
    wind_speed_eff = wind_speed * (1.-wake_deficit)

    # Calculate the power from each turbine
    # based on experienced wind speed & power curve
    # 1. By default, power output is zero
    turb_pwr = np.zeros(wake_deficit.shape)
    # 2. Determine which effective wind speeds are between cut-in and cut-out
    #    or on the curve
    between_cut_speeds = np.logical_and(turb_ci <= wind_speed_eff,
                                        wind_speed_eff < turb_co)
    below_rated = wind_speed_eff < 1.
    on_curve = np.logical_and(between_cut_speeds, below_rated)
    # 3. Between cut-in and cut-out, power is a fraction of rated
    turb_pwr[between_cut_speeds] = 1.
    # 4. Only below rated (on curve) not at 100%, but based on curve
    turb_pwr[on_curve] *= ((wind_speed_eff[on_curve] - turb_ci)
                           / (1. - turb_ci)) ** 3

    return turb_pwr
    # turb_pwr[incoming, wind directions]


def wakeless_pwr(wind_speed, turb_ci, turb_co):
    """Calculate the per-turbine wakeless power

    This is a constant and not an array as the wind speed is assumed to be
    constant.

    """
    wakeless_pwr = 0.0
    if turb_ci < wind_speed <= turb_co:
        if wind_speed >= 1:  # at or above rated
            wakeless_pwr = 1
        else:
            wakeless_pwr = ((wind_speed - turb_ci) / (1. - turb_ci)) ** 3

    return wakeless_pwr


def push_down(downwinds, deficits):
    """Return per-turbine, per-direction downwind vector"""
    gradients = np.recarray(deficits.shape, xy_pair)
    gradients.x = deficits * downwinds.x
    gradients.y = deficits * downwinds.y

    return gradients


def push_cross(downwinds, frame_vectors, deficits, blame_fractions):
    """Return per-turbine, per-direction crosswind vector"""
    # what sense should a crosswind movement go: away from the wake center
    # determine it from the sign of the crosswind frame_vector component
    crosswind_sense = np.sign(frame_vectors.c)

    cross_deficits = (deficits  # sum over all outgoing
                      * np.sum(blame_fractions * crosswind_sense, axis=0))

    gradients = np.recarray(deficits.shape, xy_pair)
    gradients.x = cross_deficits * -downwinds.y  # latter factor is crosswind.x
    gradients.y = cross_deficits * +downwinds.x  # latter factor is crosswind.y

    return gradients


def push_back(turb_vectors, deficits, blame_fractions):
    """Return per-turbine, per-direction pushback vector"""
    unit_vectors = np.recarray(turb_vectors.shape, xy_pair)
    unit_vectors.x = 0
    unit_vectors.y = 0
    dists = np.sqrt(turb_vectors.x ** 2 + turb_vectors.y ** 2)
    b = dists > 0
    unit_vectors_b = np.recarray(unit_vectors[b].shape, xy_pair)
    unit_vectors_b.x = turb_vectors[b].x / dists[b]
    unit_vectors_b.y = turb_vectors[b].y / dists[b]
    unit_vectors[b] = unit_vectors_b

    blame_deficits = np.expand_dims(deficits, 0) * blame_fractions

    gradients = np.recarray(deficits.shape, xy_pair)
    gradients.x = -np.sum(blame_deficits * np.expand_dims(unit_vectors.x, -1),
                          axis=1)  # sum over all incoming
    gradients.y = -np.sum(blame_deficits * np.expand_dims(unit_vectors.y, -1),
                          axis=1)  # sum over all incoming

    return gradients


def pseudo_gradient(wind_freq, downwinds, turb_vectors, frame_vectors,
                    deficits, blame_fractions,
                    down_coeff, cross_coeff, back_coeff):
    """Calculate the pseudo gradient for each turbine"""
    down_gradients = push_down(downwinds, deficits)
    cross_gradients = push_cross(downwinds, frame_vectors,
                                 deficits, blame_fractions)
    back_gradients = push_back(turb_vectors, deficits, blame_fractions)

    pseudo_gradients = np.recarray((deficits.shape[0],), xy_pair)
    pseudo_gradients.x = (down_gradients.x * down_coeff
                          + cross_gradients.x * cross_coeff
                          + back_gradients.x * back_coeff) @ wind_freq
    pseudo_gradients.y = (down_gradients.y * down_coeff
                          + cross_gradients.y * cross_coeff
                          + back_gradients.y * back_coeff) @ wind_freq

    return pseudo_gradients


def calcAEP(powers, wind_freq):  # powers is the output of power
    """Calculate the wind farm AEP."""

    #  Convert power to AEP
    hrs_per_year = 365. * 24.
    AEP = hrs_per_year * wind_freq * np.sum(powers, axis=0)

    return AEP


def getTurbLocYAML(file_name):
    """ Retrieve turbine locations and auxiliary file names from <.yaml> file.

    Auxiliary (reference) files supply wind rose and turbine attributes.
    """
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        props = yaml.safe_load(f)

    # Rip the coordinates (Convert from <list> to <recarray>)
    positions = props['wind_turbine_positions']
    turb_coords = np.fromiter(map(tuple, positions),
                              dtype=xy_pair, count=len(positions))
    turb_coords = turb_coords.view(np.recarray)

    # Rip the expected AEP, used for comparison
    # AEP = props['farm_output']['AEP']

    # Read the auxiliary filenames for the windrose and the turbine attributes
    fname_wr = props['wind_resource']
    fname_turb = props['wind_turbine_type']

    # Return turbine (x,y) locations, and the filenames for the others .yamls
    return turb_coords, fname_turb, fname_wr


def getWindRoseYAML(file_name):
    """Retrieve wind rose data (bins, freqs, speeds) from <.yaml> file."""
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        props = yaml.safe_load(f)

    # Rip wind directional bins, their frequency, and the farm windspeed
    # (Convert from <list> to <ndarray>)
    wind_dir = np.asarray(props['variables']['wind_direction'])
    wind_freq = np.asarray(props['probability'])
    # (Convert from <list> to <float>)
    wind_speed = float(props['variables']['wind_speed'])

    return wind_dir, wind_freq, wind_speed


def getTurbAtrbtYAML(file_name):
    '''Retreive turbine attributes from the <.yaml> file'''
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        props = yaml.safe_load(f)

    # Rip the turbine attributes
    # (Convert from <list> to <float>)
    turb_ci = float(props['cut_in_wind_speed'])
    turb_co = float(props['cut_out_wind_speed'])
    rated_ws = float(props['rated_wind_speed'])
    rated_pwr = float(props['rated_power'])
    turb_diam = float(props['rotor_diameter'])

    return turb_ci, turb_co, rated_ws, rated_pwr, turb_diam


if __name__ == "__main__":
    """Used for demonstration.

    An example command line syntax to run this file is:

        python iea37-aepcalc.py iea37-ex16.yaml

    For Python .yaml capability, in the terminal type "pip install pyyaml".
    """
    # Read necessary values from .yaml files
    # Get turbine locations and auxiliary <.yaml> filenames
    turb_coords, fname_turb, fname_wr = getTurbLocYAML(sys.argv[1])
    # Get the array wind sampling bins, frequency at each bin, and wind speed
    wind_dir, wind_freq, wind_speed = getWindRoseYAML(fname_wr)
    # Pull the needed turbine attributes from file
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML(
        fname_turb)

    # Express distances in terms of diameter lengths
    turb_coords.x /= turb_diam
    turb_coords.y /= turb_diam
    # Express speeds in terms of rated speed
    wind_speed /= rated_ws
    turb_ci /= rated_ws
    turb_co /= rated_ws
    # Express speeds in terms of rated speed
    # (nothing to do)

    # Calculate the AEP from ripped values
    vectors = turbine_vectors(turb_coords)
    downwinds = downwind_vector(wind_dir)
    frame_vectors = wind_frames(vectors, downwinds)
#    wake_deficit, blame_fractions = gaussian_wake(frame_vectors)
    wake_deficit, blame_fractions = ra_gaussian_wake(frame_vectors,
                                                     110 / turb_diam)
    powers = power(wake_deficit, wind_speed, turb_ci, turb_co)
    AEP = rated_pwr * calcAEP(powers, wind_freq)
    AEP /= 1.E3  # Convert to GWh from MWh
    # Print AEP for each binned direction, with 5 digits behind the decimal.
    print(np.array2string(AEP, precision=8, floatmode='fixed',
                          separator=', ', max_line_width=62))
    # Print AEP summed for all directions
    print(np.around(np.sum(AEP), decimals=8))
