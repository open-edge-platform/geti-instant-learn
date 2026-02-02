from getiprompt.models.grounded_sam import GroundedSAM
from getiprompt.data import Batch, Sample
from getiprompt.data.utils.image import read_image
from pathlib import Path
from getiprompt.visualizer import visualize_single_image

if __name__ == "__main__":
    img_dir = "/home/yuchunli/datasets/cat_vs_dog/test1"
    

    model = GroundedSAM()
    ref_batch = Batch.collate(
        samples=[
            Sample(categories=["cat"], category_ids=[0]),
            Sample(categories=["dog"], category_ids=[1]),
        ]
    )

    model.fit(ref_batch)

    for img_path in Path(img_dir).glob("*.jpg"):
        print(f"Processing image: {img_path}")
        img = read_image(img_path)
        sample = Sample(image=img)
        batch = Batch.collate([sample])
        predictions = model.predict(batch)
        # visualize results
        
        visualize_single_image(
            image=img,
            prediction=predictions[0],
            file_name=f"result_{img_path.stem}.png",
            color_map={0: [255, 0, 0], 1: [0, 255, 0]},
            output_folder=Path("./output"),
        )