import os, sys, random, pprint

sys.path.append('.')
import torch
import torch.optim
import torch.utils.data
from dataset.data_loader import ImagerLoader
from dataset.sampler import SequenceBatchSampler
from model.model import BaselineLSTM
from common.config import argparser
from common.logger import create_logger
from common.engine import train, validate
from common.utils import PostProcessor, get_transform, save_checkpoint, collate_fn


def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.cuda.init()

    if not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)

    logger = create_logger(args)
    logger.info(pprint.pformat(args))

    logger.info(f'Model: {args.model}')
    model = BaselineLSTM(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if not args.eval:
        train_dataset = ImagerLoader(args.img_path, args.wave_path, args.train_file, args.json_path,
                                     args.gt_path, stride=args.train_stride, transform=get_transform(True))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=SequenceBatchSampler(train_dataset, args.batch_size),
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=False)

        class_weights = torch.FloatTensor(args.weights).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(model.parameters(), args.lr)

        val_dataset = ImagerLoader(args.img_path, args.wave_path, args.val_file, args.json_path, args.gt_path,
                                   stride=args.val_stride, mode='val', transform=get_transform(False))

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=False)
    else:
        test_dataset = ImagerLoader(args.img_path, args.wave_path, args.val_file, args.json_path, args.gt_path,
                                    stride=args.test_stride, mode='val', transform=get_transform(False))

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=False)

    best_mAP = 0

    if not args.eval:
        logger.info('start training')
        for epoch in range(args.epochs):

            train_loader.batch_sampler.set_epoch(epoch)
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # TODO: implement distributed evaluation
            # evaluate on validation set
            postprocess = PostProcessor(args)
            mAP = validate(val_loader, model, postprocess)

            
            # remember best mAP in validation and save checkpoint
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            logger.info(f'mAP: {mAP:.4f} best mAP: {best_mAP:.4f}')

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'mAP': mAP},
                save_path=args.exp_path,
                is_best=is_best)


    else:
        logger.info('start evaluating')
        postprocess = PostProcessor(args)
        mAP = validate(test_loader, model, postprocess)
        logger.info(f'mAP: {mAP:.4f}')


def run():
    args = argparser.parse_args()
    main(args)


if __name__ == '__main__':
    run()
