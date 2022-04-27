python -u main.py --overwrite  --meta_model warp_leap --suffix testrun --meta_train_steps 600 --outer_kwargs 'lr' 0.3  --warp_act_fun relu &> warp_leap_testrun.600.nonlinear.olr=0.3.txt

# python -u main.py --overwrite  --meta_model warp_leap --suffix testrun --meta_train_steps 600 --outer_kwargs 'lr' 0.3 &> warp_leap_testrun.600.olr=0.3.txt
# python -u main.py --overwrite  --meta_model warp_leap_fixed --suffix testrun.fixed.3 --fixed_warp_layer_sz 3 &> warp_leap_fixed_testrun_3layers.txt
# python -u main.py --overwrite  --meta_model warp_leap_fixed --suffix testrun.fixed.3.noprewarp --fixed_warp_layer_sz 3 --no_prewarp_batchnorm &> warp_leap_fixed_testrun_3layers_nobatchnormbefore.txt

#python -u main.py --overwrite  --meta_model warp_leap_fixed --suffix testrun.fixed.2 --fixed_warp_layer_sz 3 --meta_train_steps 600 --outer_kwargs 'lr' 0.3  &> warp_leap_fixed_testrun_3layers.600.nonlinear.olr=0.3.txt
# python -u main.py --overwrite  --meta_model warp_leap_fixed --suffix testrun.fixed.2.noprewarp --fixed_warp_layer_sz 2 --no_prewarp_batchnorm --meta_train_steps 600  &> warp_leap_fixed_testrun_2layers_nobatchnormbefore.600.txt




# python -u main.py --overwrite  --meta_model warp_leap_fixed --suffix testrun.fixed.2.noprewarp --fixed_warp_layer_sz 2 --no_prewarp_batchnorm --meta_train_steps 300
# [task avg ] time:1.732 train: outer=1.2061 inner=1.2297 acc=0.61 val: outer=0.9784 inner=1.1905 acc=0.62
# NPARAMS = Total Params = 0.187 M Total Warp = 0.074M Total Adapt = 0.113M


# python -u main.py --overwrite  --meta_model warp_leap_fixed --suffix testrun.fixed.2 --fixed_warp_layer_sz 2 --meta_train_steps 300 
# [task avg ] time:1.909 train: outer=1.2545 inner=1.2791 acc=0.60 val: outer=0.9449 inner=1.1595 acc=0.65
# NPARAMS  = Total Params = 0.188 M Total Warp = 0.075M Total Adapt = 0.113M

# python -u main.py --overwrite  --meta_model warp_leap --suffix testrun --meta_train_steps 300
# [task avg ] time:1.689 train: outer=1.1845 inner=1.2101 acc=0.63 val: outer=0.8238 inner=1.0158 acc=0.69
# NPARAMS =  Total Params = 0.261 M Total Warp = 0.148M Total Adapt = 0.113M




# python -u main.py --overwrite  --meta_model warp_leap_fixed --suffix testrun.fixed.2.noprewarp --fixed_warp_layer_sz 2 --no_prewarp_batchnorm --meta_train_steps 600
# [task avg ] time:1.379 train: outer=1.1247 inner=1.1462 acc=0.64 val: outer=0.8901 inner=1.1201 acc=0.66

# python -u main.py --overwrite  --meta_model warp_leap_fixed --suffix testrun.fixed.2 --fixed_warp_layer_sz 2 --meta_train_steps 600 
# [task avg ] time:1.386 train: outer=1.1926 inner=1.2171 acc=0.62 val: outer=0.8011 inner=0.9992 acc=0.69

# python -u main.py --overwrite  --meta_model warp_leap --suffix testrun --meta_train_steps 600
# [task avg ] time:1.541 train: outer=1.1159 inner=1.1437 acc=0.65 val: outer=0.7576 inner=0.9407 acc=0.71

# Outer lr = 0.3
# [task avg ] time:1.356 train: outer=1.0629 inner=1.0896 acc=0.67 val: outer=0.8367 inner=1.0323 acc=0.71
python -u main.py --overwrite  --meta_model precond --suffix testrun --meta_train_steps 600 --outer_kwargs 'lr' 0.3




python -u main.py --overwrite  --meta_model fomaml --suffix testrun --meta_train_steps 600 --outer_kwargs 'lr' 0.3
[task avg ] time:2.738 train: outer=2.1441 inner=2.1693 acc=0.31 val: outer=1.4065 inner=1.7449 acc=0.42

Our fomaml :
[task avg ] time:1.347 train: outer=2.5976 inner=2.6623 acc=0.27 val: outer=1.5289 inner=1.9090 acc=0.37

Using fomaml with our gradient packing : 
[task avg ] time:1.569 train: outer=2.5701 inner=2.6292 acc=0.26 val: outer=1.6025 inner=1.9960 acc=0.36