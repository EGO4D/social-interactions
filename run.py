import os, sys, random, pprint

sys.path.append('.')
import torch
import torch.optim
import torch.utils.data
import soundfile as sf
from tqdm import tqdm
from dataset.data_loader import ImagerLoader
from dataset.test_loader import test_ImagerLoader
from dataset.sampler import SequenceBatchSampler
from model.model import BaselineLSTM
from common.config import argparser
from common.logger import create_logger
from common.engine import train, validate, evaluate
from common.utils import PostProcessor, test_PostProcessor, get_transform, save_checkpoint, collate_fn


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
        test_dataset = test_ImagerLoader(args.test_data_path, args.seg_info, transform=get_transform(False))

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
        postprocess = test_PostProcessor(args)
        evaluate(test_loader, model, postprocess)
        postprocess.mkfile()


def run():

    args = argparser.parse_args()
    if args.eval and not os.path.exists('./seg_info.json'):
        data_dir = args.test_data_path
        seginfo = {}

        for seg_id in tqdm(os.listdir(data_dir)):
            seg_path = os.path.join(data_dir, seg_id)
            seginfo[seg_id] = {}
            frame_list = []
            for f in os.listdir(os.path.join(seg_path, 'face')):
                fid = int(f.split('.')[0])
                frame_list.append(fid)
            frame_list.sort()
            seginfo[seg_id]['frame_list'] = frame_list

            aud, sr = sf.read(os.path.join(seg_path, 'audio', 'aud.wav'))
            frame_num = int(aud.shape[0]/sr*30+1)
            seginfo[seg_id]['frame_num'] = max(frame_num, max(frame_list)+1)

        with open('./seg_info.json','w') as f:
            json.dump(seginfo, f, indent=4)
    main(args)


if __name__ == '__main__':
    run()
