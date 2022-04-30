# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================


import torch.nn as nn
import torch
import numpy as np
from mem_cnn import ConvInvertibleModel, LinearInvertibleModel
from utils import build_dict, load_state_dict, build_iterator, Res, AggRes
import pdb

# Calculates the norm of a list of vectors
def calc_norm(grads):
	norm = 0.0
	for g_ in grads:
		if g_ is not None:
			norm += (g_**2).sum()
	return np.sqrt(norm.item())


def get_grads(loss, model, vector=False):
	with torch.no_grad():
		grads = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
		if not vector:
			return grads, None
		all_flattened, shapes = [],[]
		for grad in grads:
			shapes.append(grad.shape)
			all_flattened.append(grad.view(-1))

	return torch.concat(all_flattened), shapes

def set_new_grads(model, new_grads, shapes):
	with torch.no_grad():
		cur_idx = 0
		for idx, param in enumerate(model.parameters()):
			sz = np.prod(shapes[idx])
			this_grad = new_grads[cur_idx: (cur_idx + sz)].view(shapes[idx])
			cur_idx += sz
			param.grad = this_grad

def precond_inner_step(input, output, model, optimizer, criterion, precond_model, is_last=False):
	"""Create a computation graph through the gradient operation

	Arguments:
		input (torch.Tensor): input tensor.
		output (torch.Tensor): target tensor.
		model (torch.nn.Module): task learner.
		optimizer (maml.optim): optimizer for inner loop.
		criterion (func): loss criterion.
		create_graph (bool): create graph through gradient step.
	"""
	prediction = model(input)
	loss = criterion(prediction, output)
	
	fo_gradients = None
	if is_last:
		fo_gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)

	grad_vector, shapes_ = get_grads(loss, model, vector=True)
	new_grads = precond_model(grad_vector)
	set_new_grads(model, new_grads, shapes_)

	optimizer.step()
	
	return loss, prediction, new_grads, fo_gradients

def run_dev(data_outer, device, criterion, model):
	# Run with adapted parameters on task
	val_res = Res()
	predictions = []
	loss = 0
	for i, (input, output) in enumerate(data_outer):
		input = input.to(device, non_blocking=True)
		output = output.to(device, non_blocking=True)

		prediction = model(input)
		predictions.append(prediction)

		batch_loss = criterion(prediction, output)
		loss += batch_loss

		val_res.log(batch_loss.item(), prediction, output)

	loss = loss / (i + 1)
	return loss, val_res, predictions


def precond_task(data_inner, data_outer, model, optimizer, criterion, precond_model, param_grads, precond_optimizer, meta_train=True):
	"""Adapt model parameters to task and use adapted params to predict new samples

	Arguments:
		data_inner (iterable): list of input-output for task adaptation.
		data_outer (iterable): list of input-output for task validation.
		model (torch.nn.Module): task learner.
		optimizer (maml.optim): optimizer for inner loop.
		criterion (func): loss criterion.
		create_graph (bool): create graph through gradient step.
	"""

	# Save the original model
	original_parameters = [p_.clone().detach() for p_ in model.parameters()]
	device = next(model.parameters()).device

	# Adaptation of parameters to task
	train_res = Res()
	Nb = len(data_inner)
	for i, (input, output) in enumerate(data_inner):
		input = input.to(device, non_blocking=True)
		output = output.to(device, non_blocking=True)

		
		loss, prediction, precond_model_preds, fo_gradients = precond_inner_step(input, output, model, optimizer, criterion, precond_model, is_last=((i + 1) == Nb))
		train_res.log(loss.item(), prediction, output)

		if meta_train and i < (Nb - 1):
			# Now do gradient descent for the pre-conditioner model
# 			pred_loss, _, _ = run_dev(data_outer, device, criterion, model)

			in_, out_ = data_inner[i + 1]
			in_  = in_.to(device, non_blocking=True)
			out_ = out_.to(device, non_blocking=True)
			pred_loss = criterion(model(in_), out_)

			dev_grads, _ = get_grads(pred_loss, model, vector=True)
			inverse_rate = 1.0/(precond_model.get_update_rate() + 1e-6) # Added to undo the "initial close to identity weighting"
			precond_model_preds = (-precond_model_preds * inverse_rate)
			precond_model_preds.backward(dev_grads)
# 			nn.utils.clip_grad_norm_(precond_model.parameters(), 10.0)
			# Do a gradient descent on precond here
# 			if i == (Nb - 2):
# 				with torch.no_grad():
# 					norm_ = sum([(x.grad**2).sum() for x in precond_model.parameters()]).item()
# 					norm_ = np.sqrt(norm_)
# 				pdb.set_trace()
# 				print('This is the grad norm : ', norm_)
			precond_optimizer.step()
			precond_optimizer.zero_grad()
			precond_model.update_rate_counter()

		for p in model.parameters():
			p.grad = None

	assert fo_gradients is not None, "We were unable to get the first order gradients"

	# Get the outer perf
	model.eval()
	loss, val_res, predictions = run_dev(data_outer, device, criterion, model)
	model.train()

	with torch.no_grad():
		for idx_, p_ in enumerate(model.parameters()):
			# populate the grad dictionary
			param_grads[idx_].add_(fo_gradients[idx_])

			# Update the model with the original values
			p_.copy_(original_parameters[idx_])

	return loss, predictions, (train_res, val_res)

def update_model_grads(model, grads, normalizer=1.0):
	with torch.no_grad():
		for idx, p in enumerate(model.parameters()):
			if p.grad is None:
				p.grad = torch.zeros_like(p)
			this_grad = grads[idx] * (1.0 / normalizer)
			p.grad.copy_(this_grad)

def normalize_grads(model, normalizer):
	with torch.no_grad():
		for p in model.parameters():
			if p.grad is not None:
				p.grad.div_(normalizer)


def precond_outer_step(task_iterator, model, optimizer_cls, criterion, precond_model, precond_optimizer, return_predictions=True,
					return_results=True, meta_train=True, **optimizer_kwargs):
	loss = 0
	predictions, results = [], []
	param_grads = [torch.zeros_like(p) for p in model.parameters()]

	for i, task in enumerate(task_iterator):
		inner_iterator, outer_iterator = task
		task_optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

		task_loss, task_predictions, task_res = precond_task(
			inner_iterator, outer_iterator, model, task_optimizer, criterion, precond_model, param_grads, precond_optimizer, meta_train=meta_train)

		loss += task_loss.item()

		predictions.append(task_predictions)
		results.append(task_res)


	update_model_grads(model, param_grads, normalizer=(i + 1))
	normalize_grads(precond_model, i + 1)

	loss = loss / (i + 1)
	results = AggRes(results)

	out = [loss]
	if return_predictions:
		out.append(predictions)
	if return_results:
		out.append(results)
	return out

###############################################################################


class PRECOND(nn.Module):

	"""PRECOND

	Arguments:
		model (torch.nn.Module): task learner.
		optimizer_cls (maml.optim): task optimizer. Note: must allow backpropagation through gradient steps.
		criterion (func): loss criterion.
		tensor (bool): whether meta mini-batches come as a tensor or as a list of dataloaders.
		inner_bsz (int): if tensor=True, batch size in inner loop.
		outer_bsz (int): if tensor=True, batch size in outer loop.
		inner_steps (int): if tensor=True, number of steps in inner loop.
		outer_steps (int): if tensor=True, number of steps in outer loop.

	Example:
		>>> loss = maml.forward(task_iterator)
		>>> loss.backward()
		>>> meta_optimizer.step()
		>>> meta_optimizer.zero_grad()
	"""

	def __init__(self, model, optimizer_cls, criterion, tensor,
				 inner_bsz=None, outer_bsz=None, inner_steps=None,
				 outer_steps=None,  precond_type='Linear', **optimizer_kwargs):
		super(PRECOND, self).__init__()

		self.model = model
		# TODO: LDERY - define the pre-conditioner model here
		self.optimizer_cls = optimizer_cls
		self.optimizer_kwargs = optimizer_kwargs
		self.criterion = criterion
		modelnumel = sum([p.numel() for p in model.parameters()])
		
		if precond_type == 'Linear':
			self.precond_model = LinearInvertibleModel([128, 1024], 2, modelnumel)
		else:
			self.precond_model = ConvInvertibleModel(2, 2, 4, modelnumel)

		self.precond_model.to(next(model.parameters()).device)
		self.precond_optimizer = None
		precondnumel = sum([p.numel() for p in self.precond_model.parameters()])
		print(precondnumel, modelnumel, precondnumel//modelnumel)
		print(self.precond_model)


		self.tensor = tensor
		self.inner_bsz = inner_bsz
		self.outer_bsz = outer_bsz
		self.inner_steps = inner_steps
		self.outer_steps = outer_steps

		if tensor:
			assert inner_bsz is not None, 'set inner_bsz with tensor=True'
			assert outer_bsz is not None, 'set outer_bsz with tensor=True'
			assert inner_steps is not None, 'set inner_steps with tensor=True'
			assert outer_steps is not None, 'set outer_steps with tensor=True'

	def forward(self, inputs, return_predictions=True, return_results=True, meta_train=False):
		task_iterator = inputs if not self.tensor else [
			build_iterator(i, self.inner_bsz, self.outer_bsz, self.inner_steps, self.outer_steps)
			for i in inputs]
		return precond_outer_step(
			task_iterator=task_iterator,
			model=self.model,
			optimizer_cls=self.optimizer_cls,
			criterion=self.criterion,
			precond_model=self.precond_model,
			precond_optimizer=self.precond_optimizer,
			meta_train=meta_train,
			return_predictions=return_predictions,
			return_results=return_results,
			**self.optimizer_kwargs)
