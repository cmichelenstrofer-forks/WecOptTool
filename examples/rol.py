import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt

import wecopttool as wot

plt.style.use('tableau-colorblind10')


wavefreq = 0.3 # Hz
f1 = wavefreq
nfreq = 10
freq = wot.frequency(f1, nfreq, False)

amplitude = 0.0625 # m
phase = 30 # degrees
wavedir = 0 # degrees

waves = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)



wb = wot.geom.WaveBot()
mesh_size_factor = 0.5
mesh = wb.mesh(mesh_size_factor)
fb = cpy.FloatingBody.from_meshio(mesh, name="WaveBot")

fb.add_translation_dof(name="Heave")
ndof = fb.nb_dofs

bem_data = wot.run_bem(fb, freq)

name = ["PTO_Heave",]
kinematics = np.eye(ndof)
controller = None
loss = None
pto_impedance = None
pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)

f_add = {'PTO': pto.force_on_wec}

wec = wot.WEC.from_bem(bem_data, f_add=f_add)



obj_fun = pto.mechanical_average_power
nstate_opt = 2*nfreq


scale_x_wec = 1 # 1e1
scale_x_opt = 1 # 1e-3
scale_obj = 1 # 1e-2

results = wec.solve(
    waves,
    obj_fun,
    nstate_opt,
    scale_x_wec=scale_x_wec,
    scale_x_opt=scale_x_opt,
    scale_obj=scale_obj,
    # parameter_file='parameters.xml',
    )

opt_mechanical_average_power = results[0]['f']
print(f'Optimal average mechanical power: {opt_mechanical_average_power} W')
