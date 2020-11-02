import torch
from deep_privacy import config
from deep_privacy.engine import Trainer, ProgressiveTrainer
# Debug
parser = config.default_parser()
parser.add_argument(
    "--debug", default=False, action="store_true")
args = parser.parse_args()

if args.debug:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(precision=10)

cfg = config.Config.fromfile(args.config_path)
cfg.dump()
if cfg.trainer.progressive.enabled:
    trainer = ProgressiveTrainer(cfg)
else:
    trainer = Trainer(cfg)
trainer.train()
