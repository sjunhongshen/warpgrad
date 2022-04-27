import torch
import torch.nn as nn
import memcnn
import pdb

def make_bias_unlearnable(layer):
	for n, p in layer.named_parameters():
			if 'bias' in n:
				p.requires_grad = False

class InvertibleModel(nn.Module):
	def __init__(self, innerlayers, outerlayers, nchannels, nmodelparams):
		super(InvertibleModel, self).__init__()
		self.nchannels = nchannels
		self.nouterlayers = outerlayers
		for o_id in range(outerlayers):
			invertible_module = memcnn.AffineCoupling(
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
