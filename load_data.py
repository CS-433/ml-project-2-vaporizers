import os
import numpy as np

def load_data():

	# Step 1: load the bases

	fields = {'velocity', 'pressure', 'displacement'}

	basis_space, sv_space, Nh_space, nmodes_space = dict(), dict(), dict(), dict()
	basis_time, sv_time, Nh_time, nmodes_time = dict(), dict(), dict(), dict()
	nmodes = dict()
	for field in fields:
		basis_space[field] = np.load(os.path.join('basis', field, 'space_basis.npy'))
		sv_space[field] = np.load(os.path.join('basis', field, 'space_sv.npy'))
		Nh_space[field], nmodes_space[field] = basis_space[field].shape
		basis_time[field] = np.load(os.path.join('basis', field, 'time_basis.npy'))
		sv_time[field] = np.load(os.path.join('basis', field, 'time_sv.npy'))
		Nh_time[field], nmodes_time[field] = basis_time[field].shape
		nmodes[field] = nmodes_space[field] * nmodes_time[field]

	# UPDATE VELOCITY BASES TO ACCOUNT FOR SUPREMIZERS AND STABILIZERS
	N_supr_space = basis_space['pressure'].shape[1] + 63  # number of extra bases in space for the velocity, needed for stability
	N_supr_time = 12  # number of extra bases in time for the velocity, needed for stability

	nmodes_space['velocity_full'] = nmodes_space['velocity']
	nmodes_time['velocity_full'] = nmodes_time['velocity']
	nmodes['velocity_full'] = nmodes['velocity']

	nmodes_space['velocity'] -= N_supr_space
	nmodes_time['velocity'] -= N_supr_time
	nmodes['velocity'] = nmodes_space['velocity'] * nmodes_time['velocity']

	nmodes_space['displacement'] = nmodes_space['velocity']
	nmodes_time['displacement'] = nmodes_time['velocity']
	nmodes['displacement'] = nmodes['velocity']

	basis_space['velocity'] = basis_space['velocity'][:, :nmodes_space['velocity']]
	basis_time['velocity'] = basis_time['velocity'][:, :nmodes_time['velocity']]
	basis_space['displacement'] = basis_space['velocity']
	Nh_space['displacement'] = Nh_space['velocity']
	basis_time['displacement'] = basis_time['displacement'][:, :nmodes_time['displacement']]

	# Step 2: load the solutions

	n_snaps = None
	_sol = np.load(os.path.join('RB_data', 'solutions.npy'))[:n_snaps]

	solutions = dict()

	solutions['velocity_full'] = np.reshape(_sol[:, :nmodes['velocity_full']],
											(-1, nmodes_space['velocity_full'], nmodes_time['velocity_full']))
	solutions['velocity'] = solutions['velocity_full'][:, :nmodes_space['velocity'], :nmodes_time['velocity']]

	solutions['pressure'] = np.reshape(_sol[:, :nmodes['pressure']],
									(-1, nmodes_space['pressure'], nmodes_time['pressure']))

	solutions['displacement'] = solutions['velocity']

	# Step 3: load the parameters

	params = np.load(os.path.join('RB_data', 'parameters.npy'))
	params = np.delete(params, 2, axis=1)

	return params, solutions, basis_space, basis_time, sv_space, sv_time


def project(sol, basis_space, basis_time):
    """ Project a full-order solution in space-time."""

    return (basis_space.T.dot(sol)).dot(basis_time)


def expand(sol, basis_space, basis_time):
    """ Expand a reduced-order solution in space-time."""

    return (basis_space.dot(sol)).dot(basis_time.T)