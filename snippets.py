import yaml
import numpy as np
import IEA37_wflocs_basis as wflocs
import IEA37_wflocs_optimizers as optz
import IEA37_wflocs_plotters as pltr

# import windroses
with open('IEA37-wind_resource.case_study.yaml') as f:
    props = yaml.safe_load(f)
wind_dirs = np.array(props['variables']['wind_direction'])
wind_freq = np.array(props['probability'])

downwind_vectors = wflocs.downwind_vector(wind_dirs)

# plotting windroses
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.set_aspect(1.0)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.set_ylim(0,0.225)

ax.bar(oldpmf[:, 0] / 360 * 2 * pi, oldpmf[:, 1], color='b', width=2*pi/16)
savefig('windrose.mean.pdf', bbox_inches='tight')

ax.bar(pmf[:, 0] / 360 * 2 * pi, pmf[:, 1], color='r', width=2*pi/16)
savefig('windrose.variant.pdf', bbox_inches='tight')

ax.bar(meanpmf[:, 0] / 360 * 2 * pi, meanpmf[:, 1], color='g', width=2*pi/16)
savefig('windrose.mean.pdf', bbox_inches='tight')

# importing initial layout
with open('initial_layout.yaml', 'r') as f:
    props = yaml.safe_load(f)
positions = props['wind_turbine_positions']
turb_coords = np.fromiter(map(tuple, positions),
                          dtype=wflocs.coordinate,
                          count=len(positions))
turb_coords = turb_coords.view(np.recarray)

turb_coords.x /= 130
turb_coords.y /= 130
layout = np.copy(turb_coords).view(np.recarray)

# plot layout
pltr.layout_scatter(layout, layout, 2000 / 130)

# import parameters
wind_speed = 9.8
turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = wflocs.getTurbAtrbtYAML('wind_turbine.yaml')
wind_speed /= rated_ws
turb_ci /= rated_ws
turb_co /= rated_ws

# optimize layout
new_layout = optz.search_equilibrium(layout, 2000 / 130, 100, downwind_vectors, wind_freq, wind_speed, turb_ci, turb_co)