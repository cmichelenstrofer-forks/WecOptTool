import autograd.numpy as np
import capytaine as cpy
import matplotlib.pyplot as plt

import wecopttool as wot
# import pyrol as rol
# import pyrol.vectors

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

def const_f_pto(wec, x_wec, x_opt, waves): # Format for scipy.optimize.minimize
    f_max = 750.0
    nsubsteps = 1 # need to allow larger
    f = pto.force_on_wec(wec, x_wec, x_opt, waves, nsubsteps)
    return f_max - np.abs(f.flatten())

# constraints = []
constraints =  [const_f_pto,]


wec = wot.WEC.from_bem(bem_data, f_add=f_add, constraints=constraints)



obj_fun = pto.mechanical_average_power
nstate_opt = 2*nfreq

results = wec.solve(waves, obj_fun, nstate_opt)

opt_mechanical_average_power = results[0]['f']
print(f'Optimal average mechanical power: {opt_mechanical_average_power} W')
