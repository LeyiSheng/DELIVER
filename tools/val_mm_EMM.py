import torch
import argparse
import yaml
import math
import os
import time
import itertools
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou

# Optional imports for CAFuser integration (Detectron2/OneFormer-based)
_CAFUSER_AVAILABLE = False
try:
    import sys
    from pathlib import Path as _Path
    # Try to add CAFuser to the path if sibling repo exists
    _this_file = _Path(__file__).resolve()
    _cafuser_root = (_this_file.parents[2] / 'CAFuser')
    if _cafuser_root.exists():
        sys.path.insert(0, str(_cafuser_root))
    # Detectron2 / OneFormer imports (registered by CAFuser)
    from detectron2.config import get_cfg as d2_get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model as d2_build_model
    # CAFuser/OneFormer config adders
    from oneformer import add_oneformer_config, add_common_config, add_swin_config, add_dinat_config, add_convnext_config
    from cafuser.config import add_cafuser_config, add_deliver_config
    _CAFUSER_AVAILABLE = True
except Exception:
    # We will only error if user actually selects MODEL.NAME == 'CAFUSER'
    _CAFUSER_AVAILABLE = False

class CAFUSERWrapper(torch.nn.Module):
    """Minimal wrapper to run CAFuser inside DELIVER eval loop.

    Expects list of modality tensors [B, C, H, W] in the order given by
    cfg['DATASET']['MODALS'] (e.g., ['img','depth','event','lidar']).
    Produces tensor [B, num_classes, H, W] of per-class probabilities.
    """
    MODALITY_MAP = {
        'img': 'CAMERA',
        'image': 'CAMERA',
        'rgb': 'CAMERA',
        'depth': 'DEPTH',
        'event': 'EVENT',
        'lidar': 'LIDAR',
    }

    def __init__(self, caf_cfg, d2_model, deliver_modals: list):
        super().__init__()
        self.caf_cfg = caf_cfg
        self.model = d2_model
        self.deliver_modals = deliver_modals
        # CAFuser modality order and main modality from its config
        try:
            self.caf_mod_order = [m.upper() for m in caf_cfg.DATASETS.DELIVER.MODALITIES.ORDER]
        except Exception:
            # Fallback to default
            self.caf_mod_order = ['CAMERA', 'LIDAR', 'EVENT', 'DEPTH']
        try:
            self.main_mod = caf_cfg.DATASETS.DELIVER.MAIN_MODALITY.upper()
        except Exception:
            self.main_mod = 'CAMERA'

        # Build reverse index from CAFuser modality name -> index in deliver input list
        lower_deliver = [m.lower() for m in deliver_modals]
        self.caf_to_idx = {}
        for caf_mod in self.caf_mod_order:
            # map CAF modality to expected lower name in deliver list
            # e.g., CAMERA -> img, DEPTH -> depth, EVENT -> event, LIDAR -> lidar
            target_lower = None
            for k, v in CAFUSERWrapper.MODALITY_MAP.items():
                if v == caf_mod:
                    target_lower = k
                    break
            if target_lower is None:
                raise ValueError(f"Unknown CAFuser modality mapping for {caf_mod}")
            if target_lower not in lower_deliver:
                raise ValueError(f"Input modalities {deliver_modals} missing required CAFuser modality {caf_mod}")
            self.caf_to_idx[caf_mod] = lower_deliver.index(target_lower)

    @torch.no_grad()
    def forward(self, images_list: list):
        # images_list: list of modality tensors, each [B,C,H,W]
        B = images_list[0].shape[0]
        H, W = images_list[0].shape[2], images_list[0].shape[3]

        batched_inputs = []
        for b in range(B):
            record = {
                'modalities': self.caf_mod_order,
                'height': H,
                'width': W,
                'task': 'The task is semantic',
            }
            # Assign per-modality tensors
            for caf_mod in self.caf_mod_order:
                idx = self.caf_to_idx[caf_mod]
                tensor_bc = images_list[idx][b]  # [C,H,W]
                record[caf_mod] = tensor_bc
                if caf_mod == self.main_mod:
                    record['image'] = tensor_bc
            batched_inputs.append(record)

        outputs = self.model(batched_inputs)
        # outputs is a list of dicts, each with key 'sem_seg': [num_classes, H, W]
        out = []
        for o in outputs:
            if 'sem_seg' not in o:
                raise RuntimeError('CAFuser model did not return semantic output')
            out.append(o['sem_seg'])
        return torch.stack(out, dim=0)

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

@torch.no_grad()
def sliding_predict(model, image, num_classes, flip=True):
    image_size = image[0].shape
    tile_size = (int(ceil(image_size[2]*1)), int(ceil(image_size[3]*1)))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((num_classes, image_size[2], image_size[3]), device=torch.device('cuda'))
    count_predictions = torch.zeros((image_size[2], image_size[3]), device=torch.device('cuda'))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = [modal[:, :, y_min:y_max, x_min:x_max] for modal in image]
            padded_img = [pad_image(modal, tile_size) for modal in img]
            tile_counter += 1
            padded_prediction = model(padded_img)
            if flip:
                fliped_img = [padded_modal.flip(-1) for padded_modal in padded_img]
                fliped_predictions = model(fliped_img)
                padded_prediction += fliped_predictions.flip(-1)
            predictions = padded_prediction[:, :, :img[0].shape[2], :img[0].shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.squeeze(0)

    return total_predictions.unsqueeze(0)

@torch.no_grad()
def evaluate(model, dataloader, device):
    print('Evaluating...')
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    sliding = False
    for images, labels in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        if sliding:
            preds = sliding_predict(model, images, num_classes=n_classes).softmax(dim=1)
        else:
            preds = model(images).softmax(dim=1)
        metrics.update(preds, labels)
    
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, f1, mf1, ious, miou


@torch.no_grad()
def evaluate_with_missing_modalities(model, dataloader, device):
    print('Evaluating with missing modalities...')
    model.eval()
    n_classes = dataloader.dataset.n_classes
    
    
    # Define the modal order and the total number of modalities
    modalities = ['RGB', 'D', 'E', 'L']
    n_modalities = len(modalities)

    # Store results for each modality combination
    results = {}

    # Iterate over all combinations of missing modalities
    for num_missing in range(n_modalities):
        for missing_modalities in itertools.combinations(modalities, num_missing):
            metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
            print(f'Evaluating with missing modalities: {missing_modalities}')
            
            for images, labels in tqdm(dataloader):
                images = [x.to(device) for x in images]
                labels = labels.to(device)
                

                # Zero out the missing modalities
                for modality in missing_modalities:
                    index = modalities.index(modality)
                    images[index].zero_()  # Set the corresponding modality to zero
                
                '''print(f'Images after zeroing out modalities: {missing_modalities}')
                for modality in missing_modalities:
                    index = modalities.index(modality)
                    print(f'Modality {modality} has been set to zero. Image shape: {images[index].shape}, Zeroed pixels: {images[index].numel()}')'''
                # Make predictions
                # CMNeXt：preds = model(images).softmax(dim=1)
                # GeminiFusion preds = model(images)
                preds = model(images)
                
                if cfg["MODEL"]["NAME"] == "GeminiFusion":
                    metrics.update(preds[-1], labels)
                elif cfg["MODEL"]["NAME"] == "stitchfusion":
                    metrics.update(preds.softmax(dim=1), labels)
                else:
                    metrics.update(preds, labels)
                # No need to reset images, as we are using `images` directly from the dataloader
                # and they will be reloaded in the next iteration.

            # Compute metrics
            ious, miou = metrics.compute_iou()
            acc, macc = metrics.compute_pixel_acc()
            f1, mf1 = metrics.compute_f1()

            # Store results
            results[missing_modalities] = {
                'acc': acc,
                'macc': macc,
                'f1': f1,
                'mf1': mf1,
                'ious': ious,
                'miou': miou
            }
            print(f'Results for missing modalities {missing_modalities}:')
            print(f'mIoU: {miou}')

    return results

@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = [F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=True) for img in images]
            scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    # cases = ['cloud', 'fog', 'night', 'rain', 'sun']
    # cases = ['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    cases = [None] # all
    
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): 
        raise FileNotFoundError
    print(f"Evaluating {model_path}...")

    exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    eval_path = os.path.join(os.path.dirname(eval_cfg['MODEL_PATH']), 'eval_{}.txt'.format(exp_time))

    for case in cases:
        dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, cfg['DATASET']['MODALS'], case)
        # --- test set
        # dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'test', transform, cfg['DATASET']['MODALS'], case)

        # Build model
        if cfg['MODEL']['NAME'].lower() == 'cafuser':
            if not _CAFUSER_AVAILABLE:
                raise ImportError('CAFUSER evaluation requires Detectron2 + OneFormer + CAFuser to be installed and discoverable on PYTHONPATH.')
            caf_cfg_path = eval_cfg.get('CAFUSER_CFG', None)
            if not caf_cfg_path:
                raise ValueError("EVAL.CAFUSER_CFG is required when MODEL.NAME == 'CAFUSER'")
            # Build Detectron2 cfg
            caf_cfg = d2_get_cfg()
            add_common_config(caf_cfg)
            add_oneformer_config(caf_cfg)
            add_swin_config(caf_cfg)
            add_dinat_config(caf_cfg)
            add_convnext_config(caf_cfg)
            add_cafuser_config(caf_cfg)
            add_deliver_config(caf_cfg)
            caf_cfg.merge_from_file(caf_cfg_path)
            # Ensure semantic inference is enabled
            try:
                caf_cfg.MODEL.TEST.SEMANTIC_ON = True
                caf_cfg.MODEL.TEST.INSTANCE_ON = False
                caf_cfg.MODEL.TEST.PANOPTIC_ON = False
            except Exception:
                pass
            # Build and load weights
            d2_model = d2_build_model(caf_cfg)
            DetectionCheckpointer(d2_model).load(str(model_path))
            d2_model.eval()
            d2_model.to(device)
            # Wrap to align interfaces
            model = CAFUSERWrapper(caf_cfg, d2_model, cfg['DATASET']['MODALS']).to(device)
        else:
            model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes, cfg['DATASET']['MODALS'])
            msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu'),strict = False)
            print(msg)
            model = model.to(device)
        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=eval_cfg['BATCH_SIZE'], pin_memory=False, sampler=sampler_val)
        '''if True:
            if eval_cfg['MSF']['ENABLE']:
                acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
            else:
                acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device)

            table = {
                'Class': list(dataset.CLASSES) + ['Mean'],
                'IoU': ious + [miou],
                'F1': f1 + [mf1],
                'Acc': acc + [macc]
            }
            print("mIoU : {}".format(miou))
            print("Results saved in {}".format(eval_cfg['MODEL_PATH']))

        with open(eval_path, 'a+') as f:
            f.writelines(eval_cfg['MODEL_PATH'])
            f.write("\n============== Eval on {} {} images =================\n".format(case, len(dataset)))
            f.write("\n")
            print(tabulate(table, headers='keys'), file=f)'''

        if True:
            # Check if Multi-Scale Fusion is enabled
            if eval_cfg['MSF']['ENABLE']:
                acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
            else:
                # Evaluate with missing modalities
                results = evaluate_with_missing_modalities(model, dataloader, device)
                # Initialize variables for expectation calculations
                expect_p1_iou = 0
                expect_p1_f1 = 0
                expect_p1_acc = 0                
                expect_p2_iou = 0
                expect_p2_f1 = 0
                expect_p2_acc = 0
                expect_p3_iou = 0
                expect_p3_f1 = 0
                expect_p3_acc = 0
                total_probability_p1 = 0
                total_probability_p2 = 0
                total_probability_p3 = 0
                # Prepare data for the table
                table = {
                    'Missing Modalities': [],
                    'IoU': [],
                    'F1': [],
                    'Acc': []
                }

                # Populate the table with results for each combination of missing modalities
                for missing_modalities, metrics in results.items():
                    table['Missing Modalities'].append(str(missing_modalities))
                    table['IoU'].append(metrics['miou'])
                    table['F1'].append(metrics['mf1'])
                    table['Acc'].append(metrics['macc'])

                    # Calculate the probability of each missing modality
                    # Assuming p is the probability for a single modality being missing
                    p1 = 0.2  
                    p2 = 0.1
                    p3 = 0.05# Replace with the actual probability value
                    num_missing = len(missing_modalities)  # This assumes missing_modalities is iterable
                    N = 4  # Replace with the actual total number of modalities
                    probability_p1 = (p1 ** num_missing) * ((1 - p1) ** (N - num_missing))
                    probability_p2 = (p2 ** num_missing) * ((1 - p2) ** (N - num_missing))
                    probability_p3 = (p3 ** num_missing) * ((1 - p3) ** (N - num_missing))  # Probability of this specific combination of missing modalities

                    # Update the total probability for normalization later
                    total_probability_p1 += probability_p1
                    total_probability_p2 += probability_p2
                    total_probability_p3 += probability_p3
                        
                    # Update expectation values
                    expect_p1_iou += metrics['miou'] * probability_p1
                    expect_p1_f1 += metrics['mf1'] * probability_p1
                    expect_p1_acc += metrics['macc'] * probability_p1
                    
                    expect_p2_iou += metrics['miou'] * probability_p2
                    expect_p2_f1 += metrics['mf1'] * probability_p2
                    expect_p2_acc += metrics['macc'] * probability_p2

                    expect_p3_iou += metrics['miou'] * probability_p3
                    expect_p3_f1 += metrics['mf1'] * probability_p3
                    expect_p3_acc += metrics['macc'] * probability_p3


                # Calculate final expectations by normalizing with total probability
                if total_probability_p1 > 0:
                    expect_p1_iou /= total_probability_p1
                    expect_p1_f1 /= total_probability_p1
                    expect_p1_acc /= total_probability_p1 

                if total_probability_p2 > 0:
                    expect_p2_iou /= total_probability_p2
                    expect_p2_f1 /= total_probability_p2
                    expect_p2_acc /= total_probability_p2

                if total_probability_p3 > 0:
                    expect_p3_iou /= total_probability_p3
                    expect_p3_f1 /= total_probability_p3
                    expect_p3_acc /= total_probability_p3

                # Print the overall mean for IoU, F1, and Acc as separate entries in the table
                mean_iou = sum(metrics['miou'] for metrics in results.values()) / len(results)
                mean_f1 = sum(metrics['mf1'] for metrics in results.values()) / len(results)
                mean_acc = sum(metrics['macc'] for metrics in results.values()) / len(results)

                table['Missing Modalities'].append('Mean')
                table['IoU'].append(mean_iou)
                table['F1'].append(mean_f1)
                table['Acc'].append(mean_acc)

                # Append expectations to the table
                table['Missing Modalities'].append('Expectation_p0.2')
                table['IoU'].append(expect_p1_iou)
                table['F1'].append(expect_p1_f1)
                table['Acc'].append(expect_p1_acc)

                table['Missing Modalities'].append('Expectation_p0.1')
                table['IoU'].append(expect_p2_iou)
                table['F1'].append(expect_p2_f1)
                table['Acc'].append(expect_p2_acc)

                table['Missing Modalities'].append('Expectation_p0.05')
                table['IoU'].append(expect_p3_iou)
                table['F1'].append(expect_p3_f1)
                table['Acc'].append(expect_p3_acc)

                print("Results saved in {}".format(eval_cfg['MODEL_PATH']))

            # Save the evaluation results to a file
            with open(eval_path, 'a+') as f:
                f.writelines(eval_cfg['MODEL_PATH'])
                f.write("\n============== Eval on {} {} images =================\n".format(case, len(dataset)))
                f.write("\n")
                print(tabulate(table, headers='keys'), file=f)

            # Print the table to the console
            print(tabulate(table, headers='keys'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/DELIVER.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    # gpu = setup_ddp()
    # main(cfg, gpu)
    main(cfg)
