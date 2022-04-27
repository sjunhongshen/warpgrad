"""Runtime helpers"""
# pylint: disable=invalid-name,too-many-arguments,too-many-instance-attributes
import os
from os.path import join
import numpy as np
from collections import OrderedDict


def convert_arg(arg):
    """Convert string to type"""
    # pylint: disable=broad-except
    if arg.lower() == 'none':
        arg = None
    elif arg.lower() == 'false':
        arg = False
    elif arg.lower() == 'true':
        arg = True
    elif '.' in arg:
        try:
            arg = float(arg)
        except Exception:
            pass
    else:
        try:
            arg = int(arg)
        except Exception:
            pass
    return arg


def build_kwargs(args):
    """Build a kwargs dict from a list of key-value pairs"""
    kwargs = {}

    if not args:
        return kwargs

    assert len(args) % 2 == 0, "argument list %r does not appear to have key, value pairs" % args

    while args:
        k = args.pop(0)
        v = args.pop(0)
        if ':' in v:
            v = tuple(convert_arg(a) for a in v.split(':'))
        else:
            v = convert_arg(v)
        kwargs[str(k)] = v

    return kwargs


def compute_ncorrect(p, y):
    """Accuracy over a tensor of predictions"""
    _, p = p.max(1)
    correct = (p == y).sum().item()
    return correct


def compute_auc(x):
    """Compute AUC (composite trapezoidal rule)"""
    T = len(x)
    v = 0
    for i in range(1, T):
        v += ((x[i] - x[i-1]) / 2 + x[i-1]) / T
    return v


def unlink(path):
    """Unlink logfiles"""
    for f in os.listdir(path):
        f = os.path.join(path, f)
        if f.endswith('.log'):
            os.unlink(f)

###############################################################################


def write(step, meta_loss, loss, accuracy, losses, accuracies, f):
    """Write results data to file"""
    lstr = ""
    for l in losses:
        lstr += "{:f};".format(l)

    astr = ""
    for a in accuracies:
        astr += "{:f};".format(a)

    msg = "{:d},{:f},{:f},{:f},{:s},{:s}\n".format(
        step, meta_loss, loss, accuracy, lstr, astr)

    with open(f, 'a') as fo:
        fo.write(msg)



def log_status(results, idx, time):
    """Print status"""
    #pylint: disable=unbalanced-tuple-unpacking,too-many-star-expressions
    print("[{:9s}] time:{:3.3f} "
          "train: outer={:0.4f} inner={:0.4f} acc={:2.2f} "
          "val: outer={:0.4f} inner={:0.4f} acc={:2.2f}".format(
              str(idx),
              time,
              results.train_meta_loss,
              results.train_loss,
              results.train_acc,
              results.val_meta_loss,
              results.val_loss,
              results.val_acc)
          )

def write_train_res(results, step, log_dir):
    """Write results from a meta-train step to file"""
    write(step,
          results.train_meta_loss,
          results.train_loss,
          results.train_acc,
          results.train_losses,
          results.train_accs,
          join(log_dir, 'results_train_train.log'))
    write(step,
          results.val_meta_loss,
          results.val_loss,
          results.val_acc,
          results.val_losses,
          results.val_accs,
          join(log_dir, 'results_train_val.log'))


def write_val_res(results, step, case, log_dir):
    """Write task results data to file"""
    for task_id, res in enumerate(results):
        write(step,
              res.train_meta_loss,
              res.train_loss,
              res.train_acc,
              res.train_losses,
              res.train_accs,
              join(log_dir, 'results_{}_{}_train.log'.format(task_id, case)))
        write(step,
              res.val_meta_loss,
              res.val_loss,
              res.val_acc,
              res.val_losses,
              res.val_accs,
              join(log_dir, 'results_{}_{}_val.log'.format(task_id, case)))

###############################################################################


class Res:

    """Results container
    Attributes:
        losses (list): list of losses over batch iterator
        accs (list): list of accs over batch iterator
        meta_loss (float): auc over losses
        loss (float): mean loss over losses. Call ``aggregate`` to compute.
        acc (float): mean acc over accs. Call ``aggregate`` to compute.
    """

    def __init__(self):
        self.losses = []
        self.accs = []
        self.ncorrects = []
        self.nsamples = []
        self.meta_loss = 0
        self.loss = 0
        self.acc = 0

    def log(self, loss, pred, target):
        """Log loss and accuracies"""
        nsamples = target.size(0)
        ncorr = compute_ncorrect(pred.data, target.data)
        accuracy = ncorr / target.size(0)

        self.losses.append(loss)
        self.ncorrects.append(ncorr)
        self.nsamples.append(nsamples)
        self.accs.append(accuracy)

    def aggregate(self):
        """Compute aggregate statistics"""
        self.accs = np.array(self.accs)
        self.losses = np.array(self.losses)
        self.nsamples = np.array(self.nsamples)
        self.ncorrects = np.array(self.ncorrects)

        self.loss = self.losses.mean()
        self.meta_loss = compute_auc(self.losses)
        self.acc = self.ncorrects.sum() / self.nsamples.sum()


class AggRes:

    """Results aggregation container
    Aggregates results over a mini-batch of tasks
    """

    def __init__(self, results):
        self.train_res, self.val_res = zip(*results)
        self.aggregate_train()
        self.aggregate_val()

    def aggregate_train(self):
        """Aggregate train results"""
        (self.train_meta_loss,
         self.train_loss,
         self.train_acc,
         self.train_losses,
         self.train_accs) = self.aggregate(self.train_res)

    def aggregate_val(self):
        """Aggregate val results"""
        (self.val_meta_loss,
         self.val_loss,
         self.val_acc,
         self.val_losses,
         self.val_accs) = self.aggregate(self.val_res)

    @staticmethod
    def aggregate(results):
        """Aggregate losses and accs across Res instances"""
        agg_losses = np.stack([res.losses for res in results], axis=1)
        agg_ncorrects = np.stack([res.ncorrects for res in results], axis=1)
        agg_nsamples = np.stack([res.nsamples for res in results], axis=1)

        mean_loss = agg_losses.mean()
        mean_losses = agg_losses.mean(axis=1)
        mean_meta_loss = compute_auc(mean_losses)

        mean_acc = agg_ncorrects.sum() / agg_nsamples.sum()
        mean_accs = agg_ncorrects.sum(axis=1) / agg_nsamples.sum(axis=1)

        return mean_meta_loss, mean_loss, mean_acc, mean_losses, mean_accs


def consolidate(agg_res):
    """Merge a list of agg_res into one agg_res"""
    results = [sum((r.train_res, r.val_res), ()) for r in agg_res]
    return AggRes(results)


def build_dict(names, parameters):
    """Populate an ordered dictionary of parameters"""
    state_dict = OrderedDict({n: p for n, p in zip(names, parameters)})
    return state_dict

def load_state_dict(module, state_dict):
    r"""Replaces parameters and buffers from :attr:`state_dict` into
    the given module and its descendants. In contrast to the module's
    method, this function will *not* do in-place copy of underlying data on
    *parameters*, but instead replace the ``_parameter`` dict in each
    module and its descendants. This allows us to backpropr through previous
    gradient steps using the standard top-level API.

    .. note::
        You must store the original state dict (with keep_vars=True) separately
        and, when ready to update them, use :funct:`load_state_dict` to return
        as the module's parameters.

    Arguments:
        module (torch.nn.Module): a module instance whose state to update.
        state_dict (dict): a dict containing parameters and
            persistent buffers.
    """
    par_names = [n for n, _ in module.named_parameters()]

    par_dict = OrderedDict({k: v for k, v in state_dict.items() if k in par_names})
    no_par_dict = OrderedDict({k: v for k, v in state_dict.items() if k not in par_names})
    excess = [k for k in state_dict.keys()
              if k not in list(no_par_dict.keys()) + list(par_dict.keys())]

    if excess:
        raise ValueError(
            "State variables %r not in the module's state dict %r" % (excess, par_names))

    metadata = getattr(state_dict, '_metadata', None)
    if metadata is not None:
        par_dict._metadata = metadata
        no_par_dict._metadata = metadata

    module.load_state_dict(no_par_dict, strict=False)

    def load(module, prefix=''): # pylint: disable=missing-docstring
        _load_from_par_dict(module, par_dict, prefix)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)

def build_iterator(tensors, inner_bsz, outer_bsz, inner_steps, outer_steps, cuda=False, device=0):
    """Construct a task iterator from input and output tensor"""
    inner_size = inner_bsz * inner_steps
    outer_size = outer_bsz * outer_steps
    tsz = tensors[0].size(0)
    if tsz != inner_size + outer_size:
        raise ValueError(
            'tensor size mismatch: expected {}, got {}'.format(
                inner_size + outer_size, tsz))

    def iterator(start, stop, size):  #pylint: disable=missing-docstring
        for i in range(start, stop, size):
            out = tuple(t[i:i+size] for t in tensors)
            if cuda:
                out = tuple(t.cuda(device) for t in out)
            yield out

    return iterator(0, inner_size, inner_bsz), iterator(inner_size, tsz, outer_bsz)


def _load_from_par_dict(module, par_dict, prefix):
    """Replace the module's _parameter dict with par_dict"""
    _new_parameters = OrderedDict()
    for name, param in module._parameters.items():
        key = prefix + name
        if key in par_dict:
            input_param = par_dict[key]
        else:
            input_param = param

        if input_param.shape != param.shape:
            # local shape should match the one in checkpoint
            raise ValueError(
                'size mismatch for {}: copying a param of {} from checkpoint, '
                'where the shape is {} in current model.'.format(
                    key, param.shape, input_param.shape))

        _new_parameters[name] = input_param
    module._parameters = _new_parameters
