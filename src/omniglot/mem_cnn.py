import torch
import torch.nn as nn
import memcnn
import pdb

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


	def forward(self, x):
		last_idx = (x.shape[-1] // self.largest_sz)*self.largest_sz
		x, rem = torch.split(x, [last_idx, x.shape[-1] - last_idx])
		x = x.view(1, -1)

		for o_id, these_nparams in enumerate(self.nouterlayers):
			this_module = getattr(self, 'invertible_{}'.format(o_id))
			this_sz = these_nparams * 2
			x = chunk_and_run(x, this_module, this_sz)

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

		self.ln = nn.LayerNorm([nmodelparams])
		make_bias_unlearnable(self.ln)


	def forward(self, x):
		with torch.no_grad():
			old_x = x.clone().detach()
		x = x.view(1, self.nchannels, -1)
		for o_id in range(self.nouterlayers):
			this_module = getattr(self, 'invertible_{}'.format(o_id))
			x = this_module.forward(x)

		x = x.view(-1)
		x = self.ln(x)
		x = (x + old_x)/2
		return x

class InvertibleOperation(nn.Module):
	def __init__(self, nlayers,  channels, ln_channels):
		super(InvertibleOperation, self).__init__()
		blocks = []
		for _ in range(nlayers):
			blocks.append(nn.Conv1d(in_channels=channels, out_channels=channels,
											  kernel_size=9, padding=4, bias=False),)
# 			ln = nn.LayerNorm([channels, ln_channels])
# 			make_bias_unlearnable(ln)
# 			blocks.append(ln)
			blocks.append(nn.ReLU())
		blocks = blocks[:-1]
		self.seq = nn.Sequential(*blocks)

	def forward(self, x):
		return self.seq(x)
