import yaml
import numpy as np
import IEA37_wflocs_basis as wflocs
import IEA37_wflocs_optimizers as optz
import IEA37_wflocs_plotters as pltr
import matplotlib.pyplot as plt


# %%  import windroses
with open('IEA37-wind_resource.case_study.yaml') as f:
    props = yaml.safe_load(f)
wind_dirs = np.array(props['variables']['wind_direction'])
wind_freq = np.array(props['probability'])

downwind_vectors = wflocs.downwind_vector(wind_dirs)

# %% plotting windroses
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.set_aspect(1.0)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.set_ylim(0, 0.225)

ax.bar(wind_dirs / 360 * 2 * np.pi, wind_freq, color='b', width=2*np.pi/16)

# %% import parameters
rotor_diameter = 130
wind_speed = 9.8
turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = wflocs.getTurbAtrbtYAML(
        'IEA37-wind_turbine.onshore_reference.yaml')
wind_speed /= rated_ws
turb_ci /= rated_ws
turb_co /= rated_ws

# %% importing initial layout
with open('IEA37-optimization_case_study_IO.ex36.yaml', 'r') as f:
    props = yaml.safe_load(f)
site_radius = 2000
positions = props['wind_turbine_positions']
turb_coords = np.fromiter(map(tuple, positions),
                          dtype=wflocs.xy_pair,
                          count=len(positions))
turb_coords = turb_coords.view(np.recarray)
turb_coords.x /= rotor_diameter
turb_coords.y /= rotor_diameter
layout = np.copy(turb_coords).view(np.recarray)

# %% optimize layout
new_layout = optz.search_equilibrium(layout, site_radius / rotor_diameter, 100,
                                     downwind_vectors, wind_freq,
                                     wind_speed, turb_ci, turb_co)

# %% plot layout
plt.figure()
pltr.layout_scatter(layout, new_layout, site_radius / rotor_diameter)
