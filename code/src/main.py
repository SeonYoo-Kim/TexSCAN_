import argparse
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
import datasets.mvtec as mvtec
from einops import rearrange
from sklearn.neighbors import NearestNeighbors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./result")
    #parser.add_argument('--dataset_category', '-d', choices=['MVTec', 'BTAD', 'WFDD', 'WFT'], default='MVTec')
    return parser.parse_args()

def main():
    args = parse_args()
    print('pwd=', os.getcwd())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.set_printoptions(threshold=np.inf)

    # load model
    model = wide_resnet50_2(pretrained=True, progress=True)
    model.to(device)
    model.eval()
    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer2[3].register_forward_hook(hook)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:
        test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

        lable_list = []
        gt_mask_list = []
        test_imgs = []
        score_map_list = []
        scores = []
        cut_surrounding = 32

        # 이미지 단위
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test |'):
            test_imgs.extend(x.cpu().detach().numpy())
            lable_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask[:, :, cut_surrounding:x.shape[2] - cut_surrounding,
                                cut_surrounding:x.shape[2] - cut_surrounding].cpu().detach().numpy().astype(int))
                                # 상하좌우 32픽셀씩 자름. (PatchCore는 256인데 UTAD는 320인 이유)
                                # resize 없이 하려면 이거 상수로 두면 안됨
            features = get_feature(model, x, device, outputs) # return [layer2_feature]
            m = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            gallery2 = rearrange(m(features[0]), 'i c h w ->i (h w) c').unsqueeze(1).to('cpu').detach().numpy().copy()
            # einops.rearrange :  i c h w의 차원을 i (h w) c로 변경, (h w)는 곱해짐 (40*40 → 1600)
            # 이후에 1번 자리에 1차원 추가 (i, 1, 1600, c)
            # heatMap2 = calc_score(gallery2, gallery2, 0) # 각 픽셀에서 가까운 400개와의 평균거리 히트맵
            heatMap2 = calc_dbscan(gallery2, 0)

            for imgID in range(x.shape[0]):
                cut2 = 3
                newHeat = interpolate_scoremap(imgID, heatMap2, cut2, x.shape[2]) # 상하좌우 3씩 깎고 다시 보간
                # newHeat = gaussian_filter(newHeat.squeeze().cpu().detach().numpy(), sigma=4)
                # newHeat = torch.from_numpy(newHeat.astype(np.float32)).clone().unsqueeze(0).unsqueeze(0)
                # newHeat = torch.tensor(newHeat, dtype=torch.float32).clone()
                # print(f"newHeat shape: {newHeat.shape}")
                score_map_list.append(newHeat[:, :, cut_surrounding:x.shape[2]-cut_surrounding,
                                      cut_surrounding:x.shape[2] - cut_surrounding])
                scores.append(score_map_list[-1].max().item()) # 스코어맵의 최대값을 image-level 스코어로 등록?

        ##################################################
        # calculate image-level ROC AUC score
        # 클래스 단위
        fpr, tpr, _ = roc_curve(lable_list, scores) # return FPRs, TPRs, Thresholds
        roc_auc = roc_auc_score(lable_list, scores) # return roc_score
        img_log_txt = open('result/img_log.txt', 'w')
        img_log_txt.write(f"{roc_auc}\n")
        class_txt = open('result/class_name.txt', 'w')
        class_txt.write(f"{class_name}\n")
        total_roc_auc.append(roc_auc)
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))

        # calculate per-pixel level ROCAUC
        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()

        fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
        per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
        pix_log_txt = open('result/pix_log.txt', 'w')
        pix_log_txt.write(f"{per_pixel_rocauc}\n")
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # visualize localization result
        visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, class_name, 5, cut_surrounding)

        fig.tight_layout()
        fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


# def interpolate_scoremap(imgID, heatMap, cut, imgshape):
#     # blank = torch.ones_like(heatMap[imgID, :, :]) * heatMap[imgID, :, :].min()
#     blank = torch.zeros_like(heatMap[imgID, :, :])
#     blank[cut:heatMap.shape[1] - cut, cut:heatMap.shape[1] - cut] = heatMap[imgID, cut:heatMap.shape[1] - cut,
#                                                                     cut:heatMap.shape[1] - cut]
#     # 상하좌우 3씩 또 깎음
#     # return F.interpolate(blank[:, :].unsqueeze(0).unsqueeze(0), size=imgshape, mode='bininear', align_corners=False)
#     return F.interpolate(blank[:, :].unsqueeze(0).unsqueeze(0), size=imgshape, mode='nearest')

def interpolate_scoremap(imgID, heatMap, cut, imgshape):
    # 히트맵의 최소값이 0인지 확인
    blank = torch.zeros_like(heatMap[imgID, :, :])

    # 해당 영역에 히트맵 값을 복사
    blank[cut:heatMap.shape[1] - cut, cut:heatMap.shape[1] - cut] = heatMap[imgID, cut:heatMap.shape[1] - cut,
                                                                    cut:heatMap.shape[1] - cut]

    # 보간 전의 blank 값을 출력
    #print(f"Original blank (before interpolation):\n{blank}")

    # 'nearest' 보간으로 크기 조정
    interpolated_blank = F.interpolate(blank[:, :].unsqueeze(0).unsqueeze(0), size=imgshape, mode='nearest')

    # 보간 후의 값을 출력

    #print(f"Interpolated blank (after interpolation):\n{interpolated_blank.squeeze().cpu().numpy()}")

    return interpolated_blank


def get_feature(model, img, device, outputs):
    with torch.no_grad():
        _ = model(img.to(device))

    layer2_feature = outputs[-1]

    outputs.clear()
    return [layer2_feature]

def calc_dbscan(gallery, layerID):
    # gallery = (i, 1, 1600, c)
    heatmap = np.zeros((gallery.shape[0], gallery.shape[2]))  # (i, 1600)

    for img_idx in range(gallery.shape[0]):
        # 각 이미지의 갤러리에서 특정 레이어ID의 데이터를 선택합니다.
        features = gallery[img_idx, layerID, :, :]  # (1600, c)

        # DBSCAN 클러스터링을 수행합니다.
        dbscan = DBSCAN(eps=0.2, min_samples=400)  # eps와 min_samples는 데이터에 맞게 조정하세요.
        labels = dbscan.fit_predict(features)  # (1600,)

        # 라벨을 빈도수에 따라 정렬합니다.
        ranked_cluster = rank_labels(labels)

        # 라벨을 사용하여 히트맵을 생성합니다.
        for idx in range(len(labels)):
            if labels[idx] == ranked_cluster[0]:
                heatmap[img_idx, idx] = 0
            else:
                heatmap[img_idx, idx] = 1

    print(f"Initial heatmap:\n{heatmap}")
    # 히트맵을 torch 텐서로 변환합니다.
    heatmap = torch.tensor(heatmap, dtype=torch.float32).clone()

    # 히트맵을 원래 이미지의 2D 형식으로 변환합니다.
    dim = int(np.sqrt(gallery.shape[2]))  # 예를 들어, 1600이면 40x40이 됩니다.
    return heatmap.reshape(gallery.shape[0], dim, dim)  # (i, h, w)


# def calc_dbscan(gallery, layerID):
#     # gallery = (i, 1, 1600, c)
#     heatmap = np.zeros((gallery.shape[0], gallery.shape[2])) # i, 1600
#     dbscan = DBSCAN(eps=0.2, min_samples=400) # DBSCAN (eps : epsilon, min_samples : min point)
#     labels = dbscan.fit(gallery[ :, layerID, :, :])
#     ranked_cluster = rank_labels(labels)
#     for idx, i in enumerate(labels) :
#         if i == ranked_cluster[0]:
#             labels[idx] = 0
#         else :
#             labels[idx] = 1
#         heatmap[idx, :] = labels[idx]
#         heatmap = torch.from_numpy(heatmap.astype(np.float32)).clone()
#     dim = int(np.sqrt(gallery.shape[2]))
#     return heatmap.reshape(gallery.shape[0], dim, -1)

def calc_score(test, gallery, layerID):
    # test = gallery = (i, 1, 1600, c)
    heatmap = np.zeros((test.shape[0], test.shape[2])) # i, 1600
    for imgID in range(test.shape[0]):
        nbrs = NearestNeighbors(n_neighbors=400, algorithm='ball_tree').fit(gallery[imgID, layerID, :, :])
        # 1600개 모든 특징 등록
        distances, _ = nbrs.kneighbors(test[imgID, layerID, :, :])
        # 해당 imgID 특징의 knn거리 distances에 저장
        heatmap[imgID, :] = np.mean(distances, axis=1)
        heatmap = torch.from_numpy(heatmap.astype(np.float32)).clone()
    dim = int(np.sqrt(test.shape[2]))
    return heatmap.reshape(test.shape[0], dim, -1)


def rank_labels(labels):
    # 라벨의 빈도를 계산합니다.
    label_counts = Counter(labels)
    # 빈도순으로 라벨을 정렬합니다.
    sorted_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
    # 결과를 리스트로 반환합니다.
    return [label for label, count in sorted_labels]

def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, save_path, class_name, vis_num, cut_pixel):
    for t_idx in range(vis_num):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        heat = score_map_list[t_idx].flatten(0, 2).cpu().detach().numpy().copy()
        test_pred = score_map_list[t_idx].flatten(0, 2).cpu().detach().numpy()
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img =test_pred_img[cut_pixel:test_img.shape[0]-cut_pixel,cut_pixel:test_img.shape[0]-cut_pixel, :]
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(test_gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(heat)
        ax_img[2].title.set_text('HeatMap')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    main()

