from configs.default import opt
from core.trainer import IPFNTrainer
import utils.io as io

if __name__ == '__main__':
    opt = io.process_args(opt)
    trainer = IPFNTrainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()