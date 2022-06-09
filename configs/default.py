import argparse
import os
from core.app import HierarchyArgmentParser


parser = HierarchyArgmentParser()

# Experiment arguments
exp_args = parser.add_parser("experiment")
exp_args.add_argument('-n', '--experiment-id', type=str, default='debug',
                      help='experiment id')
exp_args.add_argument('--model-dir', type=str, default='trained_models', help="where to save trained model")
exp_args.add_argument('-s', '--seed', type=int, default=666, help='random seed')

# Network arguments
net_args = parser.add_parser("model")
net_args.add_argument('--input-type', type=str, default='2d',help=' 2d | 3d, which corresponds to either an image or a volume')
net_args.add_argument('--crop-res', type=float, default=256, help='either a scale or an integer, defining the size of the cropped patches')
net_args.add_argument('--source-scale', type=float, default=0.1, help='downsampling scale of the input')
net_args.add_argument('--channel-dim', type=int, default=-1, help='number of channels in the input image (image default to 3, volume default to 1')
net_args.add_argument('--latent-dim', type=int, default=64)
net_args.add_argument('--noise', type=str, default='stationary',
                      help='type of noise added to the implicit field: const | stationary')
net_args.add_argument('--noise-factor', type=float, default=1.0),
net_args.add_argument('--noise-interpolation', type=str, default='gaussian',
                      help='interpolation approach')
net_args.add_argument('--guidance-feature-type', type=str, default='none', help="type of guidance features used in conditional training: none | x | y | custom")
net_args.add_argument('--sigma', type=float, default=0.2, help='sigma for latent field interpolation')
net_args.add_argument('--k-type', type=str, default='scale', help='scale | affine')
net_args.add_argument('--k-threshold', type=float, default=-1)
net_args.add_argument('--crop-portion', type=float, default=0.5)
net_args.add_argument('--warp-noise', action='store_true')

# Dataset arguments
dataset_args = parser.add_parser("dataset")
dataset_args.add_argument('-p', '--path', type=str, default='exemplars/images/honeycombed_0003.jpg')
dataset_args.add_argument('--repeat', type=int, default=1000)
dataset_args.add_argument('--image-scale', type=float, default=0.1)
dataset_args.add_argument('--sdf-scale', type=float, default=10.0)

visdom_args = parser.add_parser("visdom")
visdom_args.add_argument( '--display-id', type=int, default=1)
visdom_args.add_argument( '--address', type=str, default="172.16.33.116") # default="http://localhost")
visdom_args.add_argument( '--port', type=int, default=8097)


# Training arguments
train_args = parser.add_parser("train")
train_args.add_argument('--device', type=str, default='cuda')
train_args.add_argument('--run-mode', type=str, default='train')
train_args.add_argument('-e', '--num-epochs', type=int, default=100,
                        help='maximum number of training epochs')
train_args.add_argument('-i', '--num-iterations', type=int, default=None,
                        help='maximum number of training iterations')
train_args.add_argument('--critic-steps', type=int, default=1,
                        help='steps to train discriminator per iteration')
train_args.add_argument('--g-steps', type=int, default=1,
                        help='steps to train generator per iteration')
train_args.add_argument('-b', '--batch-size', type=int, default=16,
                        help='batch size to train')
train_args.add_argument('--num-thread', default=8, type=int,
                        help='number of threads for loading data')
train_args.add_argument('-r','--resume-path', type=str, default=None,
                        help='Training using the pre-trained model')
train_args.add_argument('-c','--coarse-path', type=str, default=None,
                        help='Training using the pre-trained coarse model')
train_args.add_argument('--save-freq', type=int, default=1,
                        help='the frequency of saving the checkpoint (epochs)')
train_args.add_argument('--log-freq', type=int, default=4,
                        help='the frequency of logging training info (iters)')
train_args.add_argument('--eval-freq', type=int, default=1,
                        help='frequency of evaluation (epochs)')
train_args.add_argument('--shift-type', type=str, default='default', help='type of random shift applied in the training (x | y | xy | default | none)')
train_args.add_argument('--shift-factor', type=float, default=4.0, help="scale of the random shift applied in training"),
train_args.add_argument('--fix-sample', action='store_true')
train_args.add_argument('--slice', action='store_true')

# Learning rate arguments
lr_args = parser.add_parser("train_lr")
lr_args.add_argument('--init-lr', type=float, default=1e-4,
                     help='the initial learning rate')
lr_args.add_argument('--lr-type', type=str, default='exp_decay',
                     help='learning rate schedule type: exp_decay | constant')
lr_args.add_argument('--decay-rate', type=float, default=0.5,
                     help='the rate of exponential learning rate decaying')
lr_args.add_argument('--decay-step', type=int, default=1,
                     help='the frequency of exponential learning rate decaying')

# Loss funtion arguments
loss_args = parser.add_parser("train_loss")
loss_args.add_argument('--loss-type', type=str, default='soft',
                       help='type of loss function')

# Eval arguments
eval_args = parser.add_parser("eval")
eval_args.add_argument('--test-scale', type=float, default=4.0, help="scale for the synthesized pattern during inference")

# Test arguments
test_args = parser.add_parser("test")

opt = parser.parse_args()
opt.mode = opt.run_mode
opt.model.portion = opt.model.crop_portion
