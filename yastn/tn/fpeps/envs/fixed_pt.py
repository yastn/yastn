import time, logging
import torch
from collections import namedtuple


from .... import Tensor, ones, zeros, eye, YastnError, Leg, tensordot, einsum, diag
from ._env_ctm import ctm_conv_corner_spec


log = logging.getLogger("ctmrg")


def env_raw_data(env):
	''' 
	Combine all env raw tensors into a 1d tensor.
	'''
	data_list = []
	slice_list = []
	numel = 0
	for site in env.sites():
		for dirn in ["tl", "tr", "bl", "br", "t", "l", "b", "r"]:
			data_list.append(getattr(env[site], dirn)._data)
			slice_list.append((numel, len(data_list[-1])))
			numel += len(data_list[-1])

	return torch.cat(data_list), slice_list

def refill_env(env, data, slice_list):
	ind = 0
	for site in env.sites():
		for dirn in ["tl", "tr", "bl", "br", "t", "l", "b", "r"]:
			getattr(env[site], dirn)._data = torch.narrow(data, 0, *slice_list[ind])
			ind += 1

def env_T_gauge(config, T_old, T_new):
	#
	#   ----T_new--[leg1]--sigma--- == ---sigma--[leg2]---T_old ---
	#          |                                          |
	#          |                                          |

	leg1 = T_new.get_legs(axes=2)
	leg2 = T_old.get_legs(axes=0)
	T_old = T_old.transpose(axes=(2, 1, 0))

	identity1 = diag(eye(config=config, legs=(leg1, leg1.conj())))
	identity2 = diag(eye(config=config, legs=(leg2, leg2.conj())))

	M = (
		einsum(
			"ij, kl -> ikjl ",
			tensordot(T_new, T_new, axes=([0, 1], [0, 1]), conj=(0, 1)),
			identity2,
		)
		+ einsum(
			"ij, kl -> ikjl",
			identity1,
			tensordot(T_old, T_old, axes=([0, 1], [0, 1]), conj=(0, 1)),
		)
		- einsum("imj, kml -> iljk", T_new.conj(), T_old)
		- einsum("imj, kml -> jkil", T_new, T_old.conj())
	)
	M = M.fuse_legs(axes=((2, 3), (0, 1)))
	s, u = M.eigh(axes=(0, 1))
	u = u.unfuse_legs(axes=(0,))
	s_zeros = s <= 1e-8
	modes = u @ s_zeros  # set eigenvectors with non-zero eigenvalues to zero
	zero_modes = []
	# collect zero eigenvectors
	for i in range(modes.get_legs(axes=2).tD[(0,)]):
		zero_mode = zeros(
			config=config, legs=(leg1.conj(), leg2.conj())
		)  # 1 ---sigma--- 2
		non_zero = False
		for charge_sector in modes.get_blocks_charge():
			if charge_sector[-1] == 0:  # the total charge of sigma should be zero
				block = modes[charge_sector]
				if torch.norm(block[:, :, i]) > 1e-8:
					non_zero = True
					zero_mode.set_block(ts=charge_sector[:-1], val=block[:, :, i])
		if non_zero:
			zero_modes.append(zero_mode)
	if len(zero_modes) == 0:
		return None

	cs = find_coeff(zero_modes, zero_modes[0].dtype)
	zero_mode = zeros(zero_modes[0].config, legs=zero_modes[0].get_legs())
	for i in range(len(cs)):
		zero_mode += cs[i] * zero_modes[i]
	return zero_mode


@torch.enable_grad()
def find_coeff(zero_modes, dtype=torch.complex128):
	cs = torch.ones(len(zero_modes), dtype=dtype, requires_grad=True)
	optimizer = torch.optim.LBFGS(
		[cs],
		lr=1.0,
		max_iter=20,
		tolerance_grad=1e-7,
		tolerance_change=1e-9,
		line_search_fn="strong_wolfe",
	)
	# detach zero_modes from the computation graph
	for i in range(len(zero_modes)):
		zero_modes[i]._data.detach_()
	prev_loss = torch.inf

	def closure():
		start = time.time()
		optimizer.zero_grad()
		legs = zero_modes[0].get_legs()
		unitary = zeros(config=zero_modes[0].config, legs=legs)
		
		for i in range(len(zero_modes)):
			unitary = unitary + cs[i] * zero_modes[i]
		identity = diag(eye(config=zero_modes[0].config, legs=legs))
		loss = (
			tensordot(unitary, unitary, axes=(1, 1), conj=(0, 1)) - identity
		).norm() / identity.norm()
		end = time.time()
		start = time.time()
		loss.backward(retain_graph=True)
		end = time.time()
		return loss

	while True:
		optimizer.step(closure)
		loss = closure()
		print("loss:", loss.item())
		if torch.abs(loss - prev_loss) < 1e-6:
			break
		prev_loss = loss
	cs.detach_()
	return cs


def find_gauge(env_old, env):
	Gauge = namedtuple("Gauge", "t l b r")
	sigma_dict = {}
	for site in env.sites():
		sigma_list = []
		for k in ["t", "l", "b", "r"]:
			T_old = getattr(env_old[site], k)
			T_new = getattr(env[site], k)
			sigma = env_T_gauge(env.psi.config, T_old, T_new)
			if sigma is None:
				return None
			fixed_t = tensordot(
				tensordot(sigma, T_new, axes=(0, 0), conj=(1, 0)),
				sigma,
				axes=(2, 0),
			)
			print("T diff:", (fixed_t - T_old).norm() / T_old.norm())
			sigma_list.append(sigma)

		# There still exists a phase freedom for each sigma, which is fixed by the fixed-point condition of C
		for k in ["tl", "bl", "br", "tr"]:
			C_old = getattr(env_old[site], k)
			C_new = getattr(env[site], k)
			if k == "tl":
				fixed_C = tensordot(
					tensordot(sigma_list[1], C_new, axes=(0, 0), conj=(1, 0)),
					sigma_list[0],
					axes=(1, 0),
				)
				# nonzero_locs = torch.nonzero(fixed_C._data)
				nonzero_locs = torch.abs(fixed_C._data) > 1e-6
				phase = torch.mean(
					fixed_C._data[nonzero_locs] / C_old._data[nonzero_locs]
				)
				sigma_list[0] = sigma_list[0] / phase  # sigma_l -> sigma_l/phase
				fixed_C = tensordot(
					tensordot(sigma_list[1], C_new, axes=(0, 0), conj=(1, 0)),
					sigma_list[0],
					axes=(1, 0),
				)
			if k == "bl":
				fixed_C = tensordot(
					tensordot(sigma_list[2], C_new, axes=(0, 0), conj=(1, 0)),
					sigma_list[1],
					axes=(1, 0),
				)
				# nonzero_locs = torch.nonzero(fixed_C._data)
				nonzero_locs = torch.abs(fixed_C._data) > 1e-6
				phase = torch.mean(
					fixed_C._data[nonzero_locs] / C_old._data[nonzero_locs]
				)
				sigma_list[2] = (
					sigma_list[2] * phase
				)  # sigma_b^{-1} -> sigma_b^{-1}/phase
				fixed_C = tensordot(
					tensordot(sigma_list[2], C_new, axes=(0, 0), conj=(1, 0)),
					sigma_list[1],
					axes=(1, 0),
				)
			if k == "br":
				fixed_C = tensordot(
					tensordot(sigma_list[3], C_new, axes=(0, 0), conj=(1, 0)),
					sigma_list[2],
					axes=(1, 0),
				)
				# nonzero_locs = torch.nonzero(fixed_C._data)
				nonzero_locs = torch.abs(fixed_C._data) > 1e-6
				phase = torch.mean(
					fixed_C._data[nonzero_locs] / C_old._data[nonzero_locs]
				)
				sigma_list[3] = (
					sigma_list[3] * phase
				)  # sigma_r^{-1} -> sigma_r^{-1}/phase
				fixed_C = tensordot(
					tensordot(sigma_list[3], C_new, axes=(0, 0), conj=(1, 0)),
					sigma_list[2],
					axes=(1, 0),
				)
			if k == "tr":
				fixed_C = tensordot(
					tensordot(sigma_list[0], C_new, axes=(0, 0), conj=(1, 0)),
					sigma_list[3],
					axes=(1, 0),
				)
			print("C diff:", (fixed_C - C_old).norm() / C_old.norm())

		for sigma in sigma_list:
			sigma._data.detach_()
		sigma_dict[site] = Gauge(*sigma_list)

	return sigma_dict


class FixedPoint(torch.autograd.Function):
	ctm_env_out, ctm_log, t_ctm, t_check = None, None, None, None

	@staticmethod
	def fixed_point_iter(env_in, sigma_dict, opts_svd, slices, env_data):
		refill_env(env_in, env_data, slices)
		env_in.update_(
			opts_svd=opts_svd, method="2site", use_qr=False, checkpoint_move=False
		)

		for site in env_in.sites():
			for dirn in ["t", "l", "b", "r"]:
				sigma = getattr(sigma_dict[site], dirn)
				fixed_t = tensordot( 
					tensordot(sigma, getattr(env_in[site], dirn), axes=(0, 0), conj=(1, 0)),
					sigma,
					axes=(2, 0),
				)
				setattr(env_in[site], dirn, fixed_t)

		for dirn in ["tl", "bl", "br", "tr"]:
			C_new = getattr(env_in[site], dirn)
			sigma_t, sigma_l, sigma_b, sigma_r = sigma_dict[site].t, sigma_dict[site].l, sigma_dict[site].b, sigma_dict[site].r 
			if dirn == "tl":
				fixed_C = tensordot(
					tensordot(sigma_l, C_new, axes=(0, 0), conj=(1, 0)),
					sigma_t,
					axes=(1, 0),
				)
			if dirn == "bl":
				fixed_C = tensordot(
					tensordot(sigma_b, C_new, axes=(0, 0), conj=(1, 0)),
					sigma_l,
					axes=(1, 0),
				)
			if dirn == "br":
				fixed_C = tensordot(
					tensordot(sigma_r, C_new, axes=(0, 0), conj=(1, 0)),
					sigma_b,
					axes=(1, 0),
				)
			if dirn == "tr":
				fixed_C = tensordot(
					tensordot(sigma_t, C_new, axes=(0, 0), conj=(1, 0)),
					sigma_r,
					axes=(1, 0),
				)
			setattr(env_in[site], dirn, fixed_C)

		env_out_data, slices = env_raw_data(env_in)
		return env_out_data

	@torch.no_grad()
	def ctm_conv_check(env, history, corner_tol):
		converged, max_dsv, history = ctm_conv_corner_spec(env, history, corner_tol)
		print("max_dsv:", max_dsv)
		log.log(logging.INFO, f"CTM iter {len(history)} |delta_C| {max_dsv}")
		return converged, history

	def get_converged_env(
		env,
		method="2site",
		max_sweeps=100,
		iterator_step=1,
		opts_svd=None,
		corner_tol=1e-8,
	):
		t_ctm, t_check = 0.0, 0.0
		t_ctm_prev = time.perf_counter()
		converged, conv_history = False, []

		for sweep in range(max_sweeps):
			env.update_(
				opts_svd=opts_svd, method=method, use_qr=False, checkpoint_move=False
			)
			t_ctm_after = time.perf_counter()
			t_ctm += t_ctm_after - t_ctm_prev
			t_ctm_prev = t_ctm_after

			converged, conv_history = FixedPoint.ctm_conv_check(env, conv_history, corner_tol)
			if converged:
				break

		return env, converged, conv_history, t_ctm, t_check

	@staticmethod
	def forward(
		ctx, env_params, slices, yastn_config, env, opts_svd, corner_tol, ctm_args, *state_params
	):
		refill_env(env, env_params, slices)
		ctm_env_out, converged, *FixedPoint.ctm_log, FixedPoint.t_ctm, FixedPoint.t_check = FixedPoint.get_converged_env(
			env,
			max_sweeps=ctm_args.ctm_max_iter,
			iterator_step=1,
			opts_svd=opts_svd,
			corner_tol=corner_tol,
		)

		env_old = ctm_env_out.copy()
		FixedPoint.ctm_env_out = env_old
		ctm_env_out.update_(
			opts_svd=opts_svd, method="2site", use_qr=False, checkpoint_move=False
		)
			
		sigma_dict = find_gauge(env_old, ctm_env_out)
		if sigma_dict is None:
			return None, None, None, None, None

		env_out_data, slices = env_raw_data(env_old)
		ctx.save_for_backward(env_out_data)
		ctx.yastn_config = yastn_config
		ctx.ctm_args = ctm_args
		ctx.opts_svd = opts_svd
		ctx.ctm_args = ctm_args
		ctx.slices = slices
		FixedPoint.slices = slices

		ctx.env = env_old
		ctx.sigma_dict = sigma_dict

		return env_out_data

	@staticmethod
	def backward(ctx, *grad_env):
		print("Backward called")
		grads = grad_env[0]
		dA = list(grad_env)
		# dA = grads
		# Compute the whole jacobian
		# env_data = torch.utils.checkpoint.detach_variable(ctx.saved_tensors)[0] # only one element in the tuple
		# part_func = lambda env_data: FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data)
		# jac = torch.autograd.functional.jacobian(part_func, env_data)	
		for step in range(ctx.ctm_args.ctm_max_iter):
			# Compute vjp only
			env_data = torch.utils.checkpoint.detach_variable(ctx.saved_tensors)[0] # only one element in the tuple
			with torch.enable_grad():
				grads = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data), env_data, grad_outputs=grads)

			# grads = grads@jac
			for grad in grads:
				print(torch.norm(grad, p=torch.inf))
			if all([torch.norm(grad, p=torch.inf) < 1e-8 for grad in grads]):
				break
			else:
				for i in range(len(grads)):
					dA[i] = dA[i] + grads[i]
				# dA += grads

		with torch.enable_grad():
			dA = torch.autograd.grad(FixedPoint.fixed_point_iter(ctx.env, ctx.sigma_dict, ctx.opts_svd, ctx.slices, env_data), ctx.env.psi.ket.get_parameters(), grad_outputs=dA)


		return None, None, None, None, None, None, None, *dA
		
