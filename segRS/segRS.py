import os

import groundingdino.datasets.transforms as T
import numpy as np
import torch
import open_clip
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from huggingface_hub import hf_hub_download

SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

CACHE_PATH = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints"))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    model.eval()
    return model


def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


class segRS():

    def __init__(self, sam_type="vit_h",remoteclip_type = 'ViT-L-14',ckpt_path=None):
        self.sam_type = sam_type
        self.remoteclip_type = remoteclip_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_groundingdino()
        self.build_remoteclip()
        self.build_sam(ckpt_path)

    def build_sam(self, ckpt_path):
        if self.sam_type is None or ckpt_path is None:
            if self.sam_type is None:
                print("No sam type indicated. Using vit_h by default.")
                self.sam_type = "vit_h"
            checkpoint_url = SAM_MODELS[self.sam_type]
            try:
                sam = sam_model_registry[self.sam_type]()
                state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
                sam.load_state_dict(state_dict, strict=True)
            except:
                raise ValueError(f"Problem loading SAM please make sure you have the right model type: {self.sam_type} \
                    and a working checkpoint: {checkpoint_url}. Recommend deleting the checkpoint and \
                    re-downloading it.")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)
        else:
            try:
                sam = sam_model_registry[self.sam_type](ckpt_path)
            except:
                raise ValueError(f"Problem loading SAM. Your model type: {self.sam_type} \
                should match your checkpoint path: {ckpt_path}. Recommend calling SegRS \
                using matching model type AND checkpoint path")
            sam.to(device=self.device)
            self.sam = SamPredictor(sam)

    def build_groundingdino(self):
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        self.groundingdino = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)
    
    def build_remoteclip(self):
        for self.remoteclip_type in ['RN50', 'ViT-B-32', 'ViT-L-14']:
            checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{self.remoteclip_type}.pt", cache_dir='checkpoints')
            print(f'{self.remoteclip_type} is downloaded to {checkpoint_path}.')
        model_name = self.remoteclip_type # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
        cmodel, _, preprocess = open_clip.create_model_and_transforms(model_name)
        tokenizer = open_clip.get_tokenizer(model_name)
        ckpt = torch.load(os.getcwd() +"/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-L-14.pt", map_location="cpu")
        cmodel.load_state_dict(ckpt)
        cmodel = cmodel.eval()
        self.preprocess = preprocess
        self.remoteclip = cmodel
        self.tokenizer = tokenizer
        
    def predict_dino(self, image_pil, text_prompt, box_threshold, text_threshold):
        image_trans = transform_image(image_pil)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    def predict_sam(self, image_pil, boxes):
        image_array = np.asarray(image_pil)
        self.sam.set_image(image_array)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.sam.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict(self, image_pil, text_prompt, box_threshold=0.3, text_threshold=0.25):
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        masks = torch.tensor([])
        if len(boxes) > 0:
            masks = self.predict_sam(image_pil, boxes)
            masks = masks.squeeze(1)
        return masks, boxes, phrases, logits
    
    def predict_segRS(self, image_pil, text_prompt, clip_text_queries, box_threshold=0.3, text_threshold=0.25):
        
        # predicting boxes using groundingDINO
        boxes, logits, phrases = self.predict_dino(image_pil, text_prompt, box_threshold, text_threshold)
        
        # Predicting remoteCLIP scores for croped images 
        text = self.tokenizer(clip_text_queries)
        boxes_DS = [] # downsampled boxes
        box_probs = [] # probabilities of downsampled boxes given by remoteCLIP
        
        if len(boxes) > 0:
            for bbox in boxes: 
                #croping image over pridicted boxes
                c_img = image_pil.crop(bbox.numpy())
                image = self.preprocess(c_img).unsqueeze(0)
                
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = self.remoteclip.encode_image(image)
                    text_features = self.remoteclip.encode_text(text)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
                    if text_probs[0] > .70:
                        boxes_DS.append(bbox)
                        box_probs.append(torch.tensor(text_probs[0]))
        
        if len(boxes_DS) > 0:
            boxes_DS = torch.stack(boxes_DS)
            box_probs = torch.stack(box_probs) 
         
        # predicting masks for downsampled boxes 
        masks = torch.tensor([])
        if len(boxes_DS) > 0:
            masks = self.predict_sam(image_pil, boxes_DS)
            masks = masks.squeeze(1)
        return masks, boxes_DS, phrases, box_probs
    
    def cal_iou(self, mask_test,mask_gt):
        intersection = np.logical_and(mask_test, mask_gt)
        union = np.logical_or(mask_test, mask_gt)
        iou = intersection.sum()/union.sum()
        return iou

    def precision_score(self, pred_mask, gt_mask):
        intersect = np.logical_and(pred_mask, gt_mask)
        total_pixel_pred = np.sum(pred_mask)
        return intersect.sum()/total_pixel_pred.sum()

    def convert_bbox(self, box):
        x_min, y_min, box_width, box_height = box
        return [x_min, y_min, x_min+ box_width, y_min+box_height]
    
