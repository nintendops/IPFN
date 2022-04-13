from configs.default import opt
from core.trainer import BasicTrainer

if __name__ == '__main__':
    opt = io.process_args(opt)
    trainer = Trainer(opt)
    if opt.run_mode == 'train':
        trainer.train()
    else:
        trainer.eval()