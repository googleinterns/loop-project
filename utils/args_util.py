import os, argparse

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--shared",
                      type=int, default=0,
                      help="if > 0, the model will use shared resnet.")
  parser.add_argument("--reshape_to", type=int, default=32,
                      help="reshape image to this size")
  parser.add_argument("--size", type=int, default=8,
                      help="tensor size along X and Y axis")
  parser.add_argument("--aug", type=int, default=0,
                      help="whether to use data augmentation (default 1)")
  parser.add_argument("--pixel_mean", type=int, default=1,
                      help="whether to subtract the pixel mean (default 1)")
  parser.add_argument("--batch_size", type=int, default=32,
                      help="batch size (default is 32)")
  parser.add_argument("--num_blocks", type=int, default=16,
                      help="number of residual blocks")
  parser.add_argument("--num_templates", type=int, default=4,
                      help="number of templates")
  parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout (default is 0.2)")
  parser.add_argument("--in_adapter_type",
                      choices=["original", "space2depth", "strided"],
                      default="strided",
                      help="Output adapter type (default is strided)")
  parser.add_argument("--out_adapter_type",
                      choices=["v1", "v2", "isometric", "dethwise"],
                      default="isometric",
                      help="Output adapter type (default is isometric)")
  parser.add_argument("--lsmooth",
                      type=float,
                      default=0,
                      help="Label smoothing parameter (default is 0)")
  parser.add_argument("--kernel_reg",
                      type=float,
                      default=1e-5,
                      help="kernel regularization parameter (default is 1e-5)")
  parser.add_argument("--lr",
                      type=float,
                      default=2*1e-3,
                      help="learning rate (default is 0.02)")
  parser.add_argument("--data_dir",
                      type=str,
                      default='~/datasets/',
                      help="directory where the datasets will be stored")
  parser.add_argument("--num_epochs", type=int, default=200,
                      help="number of epochs")
  parser.add_argument("--num_steps", type=int, default=1500,
                      help="number of steps per epoch")
  parser.add_argument("--restore", type=int, default=0,
                      help="if > 0, the model will be restored from checkpoint.")
  parser.add_argument("--name", type=str, default="default",
                      help="experiment name.")
  parser.add_argument("--save_path", type=str, default="./experiments/",
                      help="experiments folder.")
  return parser

def write_config(args, path):
  """ writes config file. """
  results_path = os.path.join(path, "config.txt")
  with open(results_path, "w") as fl:
    for arg in vars(args):
      attr = getattr(args, arg) 
      fl.write("%s: %s\n" % (arg, attr))

