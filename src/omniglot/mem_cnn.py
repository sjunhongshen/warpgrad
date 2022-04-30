import torch
import torch.nn as nn
import memcnn
import pdb

EPSILON = 1e-8

def make_bias_unlearnable(layer):
	for n, p in layer.named_parameters():
			if 'bias' in n:
				p.requires_grad = False
				
def chunk_and_run(x, model, n_in):
	chunkz = x.shape[-1] // n_in
	these_chunks = torch.chunk(x, chunkz, dim=-1)
	result = []
	for chunk in these_chunks:
		result.append(model(chunk))
	x = torch.concat(result, dim=-1)
	return x

class LinearInvertibleModel(nn.Module):
	def __init__(self, outerlayers, innerlayers, nmodelparams):
		super(LinearInvertibleModel, self).__init__()
		self.nouterlayers = outerlayers
		self.largest_sz = 2*max(self.nouterlayers) 
		for o_id, this_nparams in enumerate(self.nouterlayers):
			invertible_module = memcnn.AdditiveCoupling(
				Fm=InvertibleLinearOperation(this_nparams, innerlayers),
				Gm=InvertibleLinearOperation(this_nparams, innerlayers),
			)
			# Initializling so we get the identity function at the start.
			invertible_module_wrapper = memcnn.InvertibleModuleWrapper(fn=invertible_module, keep_input=True, keep_input_inverse=True)
			self.add_module('invertible_{}'.format(o_id), invertible_module_wrapper)
		
		self.max_updates = 100000.0
		self.update_counter = 0
	
	def update_rate_counter(self):
		self.update_counter += 1

	def get_update_rate(self):
		rate_ = min(self.update_counter / self.max_updates, 1.0)
		return rate_

	def forward(self, x):
		last_idx = (x.shape[-1] // self.largest_sz)*self.largest_sz
		x, rem = torch.split(x, [last_idx, x.shape[-1] - last_idx])
		x = x.view(1, -1)
		original_x = x.clone().detach()

		update_rate = self.get_update_rate()
		for o_id, these_nparams in enumerate(self.nouterlayers):
			this_module = getattr(self, 'invertible_{}'.format(o_id))
			this_sz = these_nparams * 2
			this_x = chunk_and_run(x, this_module, this_sz)
			x = this_x * update_rate + (1 - update_rate) * original_x

		x = x * update_rate + (1 - update_rate) * original_x
		x = x.view(-1)
		x = torch.concat([x,rem])
		return x

class InvertibleLinearOperation(nn.Module):
	def __init__(self, n_in, nlayers):
		super(InvertibleLinearOperation, self).__init__()
		blocks = []
		for idx_ in range(nlayers):
			if idx_ == 0:
				blocks.append(nn.Linear(n_in, n_in//32, bias=False))
			else:
				blocks.append(nn.Linear(n_in//32, n_in, bias=False))
			blocks.append(nn.ReLU())
		blocks = blocks[:-1]
		self.n_in = n_in
		self.seq = nn.Sequential(*blocks)

	def forward(self, x):
		x = chunk_and_run(x, self.seq, self.n_in)
		return x


class ConvInvertibleModel(nn.Module):
	def __init__(self, innerlayers, outerlayers, nchannels, nmodelparams):
		super(ConvInvertibleModel, self).__init__()
		self.nchannels = nchannels
		self.nouterlayers = outerlayers
		for o_id in range(outerlayers):
			invertible_module = memcnn.AdditiveCoupling(
				Fm=InvertibleOperation(innerlayers, nchannels//2, nmodelparams//self.nchannels),
				Gm=InvertibleOperation(innerlayers, nchannels//2, nmodelparams//self.nchannels),
			)
			# Initializling so we get the identity function at the start.
			invertible_module_wrapper = memcnn.InvertibleModuleWrapper(fn=invertible_module, keep_input=True, keep_input_inverse=True)
			self.add_module('invertible_{}'.format(o_id), invertible_module_wrapper)

# 		self.ln = nn.LayerNorm([nmodelparams])
# 		make_bias_unlearnable(self.ln)

		self.max_updates = 200000.0
		self.update_counter = 0
	
	def update_rate_counter(self):
		self.update_counter += 1
		if self.update_counter % 5000 == 0:
			print('The current update rate is set to : ', self.get_update_rate())

	def get_update_rate(self):
		rate_ = min(self.update_counter / self.max_updates, 1.0)
		return rate_


	def forward(self, x):
		x = x.view(1, self.nchannels, -1)

		with torch.no_grad():
			old_x = x.clone().detach()
			old_norm_ = old_x.norm()

		update_rate = self.get_update_rate()
		for o_id in range(self.nouterlayers):
			this_module = getattr(self, 'invertible_{}'.format(o_id))
			x = this_module.forward(x)
			x = x * update_rate + (1 - update_rate) * old_x

		x = x.view(-1)
		with torch.no_grad():
			cur_norm_ = x.norm() + EPSILON
			scaling = (old_norm_ / cur_norm_).item()
		x = scaling * x # Update the scaling toe match the original gradient norm
		x = x  * update_rate + (1 - update_rate) * old_x.view(-1)
		return x

class InvertibleOperation(nn.Module):
	def __init__(self, nlayers,  channels, ln_channels):
		super(InvertibleOperation, self).__init__()
		blocks = []
		for l_ in range(nlayers):
			# stupid way of doing this but feeling sleepy
			if l_ == 0:
				in_, out_ = channels, channels*32
			elif l_ == (nlayers - 1):
				in_, out_ = channels*32, channels
			else:
				in_, out_ = channels*32, channels*32

			blocks.append(nn.Conv1d(in_channels=in_, out_channels=out_,
											  kernel_size=9, padding=4, bias=False),)
			blocks.append(nn.ReLU())

		blocks = blocks[:-1]
		self.seq = nn.Sequential(*blocks)

	def forward(self, x):
		return self.seq(x)
