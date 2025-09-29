import torch
from torch import nn

from onnxconverter_common import float16
from getiprompt.models.models import load_sam_model
from getiprompt.utils.constants import SAMModelName
from getiprompt.utils.utils import precision_to_torch_dtype
import onnx
import openvino
from torch.nn import functional as F



class OVSamPredictor(nn.Module):
    def __init__(self, precision="fp16"):
        super().__init__()
        sam_predictor = load_sam_model(
            SAMModelName.SAM_HQ_TINY,
            device="cpu",
            precision=precision,
            compile_models=False,
            benchmark_inference_speed=False,
            apply_onnx_patches=True,
        )
        self.model = sam_predictor.model.eval()
        self.precision = precision_to_torch_dtype(precision)

    def export(self, model_path: str):
        input_names = [
            "image", 
            "point_coords", 
            "point_labels", 
            "original_image_size"
        ]
        output_names = ["masks", "iou_predictions", "low_res_masks"]
        image = torch.zeros((1, 3, 1024, 1024), dtype=self.precision)
        point_coords = torch.rand((1, 3, 2), dtype=self.precision)  # in shape (1, N, 2)
        point_labels = torch.randint(-1, 2, (1, 3), dtype=torch.int32) #
        # mask_input = torch.randn((1, 1, 256, 256), dtype=self.precision)  # in shape (1, 3, 256, 256)
        original_image_size = torch.tensor([1024, 1024])
        
        torch.onnx.export(
            self,
            (
                image, 
                point_coords, 
                point_labels, 
                original_image_size
            ),
            model_path,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "image": {0: "batch_size"},
                "point_coords": {0: "batch_size", 1: "N"},
                "point_labels": {0: "batch_size", 1: "N"},
                "low_res_masks": {0: "batch_size", 1: "N"},
            },
        )
        onnx_model = onnx.load(model_path)
        onnx.save(onnx_model, model_path)

        exported_model = openvino.convert_model(model_path)
        for i, name in enumerate(output_names):
            traced_names = exported_model.outputs[i].get_names()
            name_found = False
            for traced_name in traced_names:
                if name in traced_name:
                    name_found = True
                    break
            name_found = name_found and bool(len(traced_names))

            if not name_found:
                msg = (
                    f"{name} is not matched with the converted model's traced output names: {traced_names}."
                    " Please check output_names argument of the exporter's constructor."
                )
                raise ValueError(msg)
            exported_model.outputs[i].tensor.set_names({name})

        for i, name in enumerate(input_names):
            traced_names = exported_model.inputs[i].get_names()
            name_found = False
            for traced_name in traced_names:
                if name in traced_name:
                    name_found = True
                    break
            name_found = name_found and bool(len(traced_names))

            if not name_found:
                msg = (
                    f"{name} is not matched with the converted model's traced input names: {traced_names}."
                    " Please check input_names argument of the exporter's constructor."
                )
                raise ValueError(msg)

            exported_model.inputs[i].tensor.set_names({name})
        save_path = model_path.replace(".onnx", ".xml")
        openvino.save_model(exported_model, save_path)

    @torch.no_grad()
    def forward(
        self, 
        image: torch.Tensor, 
        point_coords: torch.Tensor, 
        point_labels: torch.Tensor,
        original_image_size: torch.Tensor,
    ):
        input_h, input_w = image.shape[-2:]
        ori_h, ori_w = original_image_size[0], original_image_size[1]
        input_image = self.model.preprocess(image)
        features, interm_features = self.model.image_encoder(input_image)
        points = (point_coords, point_labels)

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            hq_token_only=False,
            interm_embeddings=interm_features,
        )

        # Upscale the masks to the original image resolution
        masks = F.interpolate(
            low_res_masks,
            (self.model.image_encoder.img_size, self.model.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        
        masks = masks[..., : input_h, : input_w]
        
        masks = F.interpolate(
            masks, 
            (ori_h, ori_w), 
            mode="bilinear", 
            align_corners=False
        )
        masks = masks > self.model.mask_threshold
        return masks, iou_predictions, low_res_masks


if __name__ == "__main__":
    predictor = OVSamPredictor(precision="fp32")
    predictor.export("samhq_tiny.onnx")
