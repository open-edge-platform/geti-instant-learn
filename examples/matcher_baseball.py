import glob
from segment_anything_hq import sam_model_registry, SamPredictor
import cv2
import numpy as np
from getiprompt.data import Batch, Sample
from getiprompt.models import Matcher
import torch
from torchvision.tv_tensors import Image as TVImage
from getiprompt.visualizer import Visualizer

if __name__ == "__main__":
    sam_weight = "/home/yuchunli/data/sam_hq_vit_tiny.pth"
    image_folder = "/home/yuchunli/datasets/baseball"
    
    ref_images = []
    ref_points = np.array([[1177, 2321]])
    ref_labels = np.array([1])
    target_images = []
    for i, img_path in enumerate(sorted(glob.glob(f"{image_folder}/*.jpg"))) :
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if i == 0:
            ref_images.append(img)
        else:
            target_images.append(TVImage(img.transpose(2,0,1)))

    sam_model = sam_model_registry["vit_tiny"](checkpoint=sam_weight)
    sam_model.to(device="cuda")
    predictor = SamPredictor(sam_model)

    predictor.set_image(ref_images[0])
    ref_masks, _, _ = predictor.predict(
        point_coords=ref_points,
        point_labels=ref_labels,
        multimask_output=False,
    )

    ref_batch = Batch.collate(
        samples=[
            Sample(
                image = TVImage(ref_images[0].transpose(2,0,1)), 
                masks = ref_masks,
                categories=["baseball"],
                category_ids=[0],
            )
        ]
    )

    matcher = Matcher(
        device="cuda",
    )

    visualizer = Visualizer(
        output_folder="outputs",
        class_map={0: "baseball"},
    )

    matcher.fit(reference_batch=ref_batch)
    batch_size = 4
    for i in range(0, len(target_images), batch_size):
        batch_images = target_images[i : i + batch_size]
        target_batch = Batch.collate(
            samples=[
                Sample(image = img) for img in batch_images
            ]
        )
        with torch.no_grad():
            outputs = matcher.predict(
                target_batch=target_batch,
            )
            visualizer.visualize(
                images=batch_images,
                predictions=outputs,
                file_names=[f"baseball_{i+j}.png" for j in range(len(batch_images))],
            )
