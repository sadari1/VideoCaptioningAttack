import opts
from dataloader import VideoDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    dataset = VideoDataset(opt, 'train')
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)

    # data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
    # data['labels'] = label
    # data['masks'] = mask
    # data['gts'] = gts
    # data['video_ids'] = vid_id
    for data in loader:
        for key in ('fc_feats', 'labels', 'masks', 'video_ids'):
            print("{} shape: {}".format(key, data[key].shape))
        break
