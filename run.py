import os, sys, random, pprint

sys.path.append('.')
import torch
import torch.optim
import torch.utils.data
from torch.utils.data import DistributedSampler
from dataset.data_loader import ImagerLoader
from model.model import BaselineLSTM, GazeLSTM
from common.config import argparser
from common.logger import create_logger
from common.engine import train, validate
from common.utils import PostProcessor, get_transform, save_checkpoint
from common.distributed import distributed_init, is_master, synchronize


def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.cuda.init()

    if args.dist:
        distributed_init(args)

    if is_master() and not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)
    synchronize()

    logger = create_logger(args)
    logger.info(pprint.pformat(args))

    logger.info(f'Model: {args.model}')
    model = eval(args.model)(args)

    if args.dist:
        model.to(args.device_id)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.rank],
            output_device=args.rank,
            find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not args.eval:
        train_dataset = ImagerLoader(args.source_path, args.train_file, args.json_path, 
                                    args.gt_path, stride=args.train_stride, transform=get_transform(True))

        val_dataset = ImagerLoader(args.source_path, args.val_file, args.json_path, args.gt_path, 
                                stride=args.val_stride, mode='val', transform=get_transform(False))

        params = {}
        if args.dist:
            params = {'sampler':DistributedSampler(train_dataset)}
        else:
            params = {'shuffle': True}

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers, 
            pin_memory=False, 
            **params)
        
        if args.dist:
            params = {'sampler':DistributedSampler(val_dataset, shuffle=False)}
        else:
            params = {'shuffle': False}

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers, 
            pin_memory=False, 
            **params)
    else:
        test_dataset = ImagerLoader(args.source_path, args.val_file, args.json_path, args.gt_path,
                                    stride=args.test_stride, mode='test', transform=get_transform(False))
        
        if args.dist:
            params = {'sampler': DistributedSampler(test_dataset, shuffle=False)}
        else:
            params = {'shuffle': False}

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            **params)

    class_weights = torch.FloatTensor(args.weights).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    best_mAP = 0

    synchronize()

    if not args.eval:
        logger.info('start training')
        for epoch in range(args.epochs):
            if args.dist:
                train_loader.sampler.set_epoch(epoch)
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            postprocess = PostProcessor(args)
            mAP = validate(val_loader, model, postprocess)

            if is_master():
                # remember best mAP in validation and save checkpoint
                is_best = mAP > best_mAP
                best_mAP = max(mAP, best_mAP)
                logger.info(f'mAP: {mAP:.4f} best mAP: {best_mAP:.4f}')

                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'mAP': mAP},
                    save_path=args.exp_path,
                    is_best=is_best,
                    is_dist=args.dist)

            synchronize()
    else:
        logger.info('start evaluating')
        postprocess = PostProcessor(args)
        mAP = validate(test_loader, model, postprocess)
        if is_master():
            logger.info(f'mAP: {mAP:.4f}')


def distributed_main(device_id, args):
    args.rank = args.start_rank + device_id
    args.device_id = device_id
    main(args)


def run():
    args = argparser.parse_args()

    if args.dist:
        args.world_size = max(1, torch.cuda.device_count())
        assert args.world_size <= torch.cuda.device_count()

        if args.world_size > 0 and torch.cuda.device_count() > 1:
            port = random.randint(10000, 20000)
            args.init_method = f"tcp://localhost:{port}"
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args,),
                nprocs=args.world_size,
            )
    else:
        main(args)


if __name__ == '__main__':
    run()
