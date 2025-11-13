import torch
from getiprompt.models import GroundedSAM
from getiprompt.data import LVISDataset

from torch.utils.data import DataLoader
from getiprompt.data.base.batch import Batch

from matplotlib import pyplot as plt

def main():
    """Main function to run the Matcher example."""
    # Define file paths
    dataset = LVISDataset(
        root="/home/yuchunli/datasets/lvis",
        n_shots=1,
        categories=['doughnut', 'cupcake', 'pastry'],
    )

    reference_dataset = dataset.get_reference_dataset()

    reference_batch = Batch.collate(reference_dataset)

    # Initialize the Matcher model
    model = GroundedSAM(
        precision="bf16",
        compile_models=False,
        benchmark_inference_speed=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model.learn(reference_batch)
    
    target_dataset = dataset.get_target_dataset()

    target_dataloader = DataLoader(
        target_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=Batch.collate,
    )
    
    # visualize results
    categories = dataset.categories
    for batch in target_dataloader:
        results = model.infer(batch)
        for img, masks in zip(batch.images, results.masks, strict=False):
            _masks = masks.data
            # make column grid of masks
            col = 5
            row = (len(_masks) + col - 1) // col
            fig, axes = plt.subplots(row, col, figsize=(10, 10))
            for i, (class_id, mask) in enumerate(_masks):
                axes[i // col, i % col].imshow(mask)
                axes[i // col, i % col].set_title(categories[class_id])
            plt.savefig(f"results/grounded_sam_{i}.png")
            plt.close()

if __name__ == "__main__":
    main()