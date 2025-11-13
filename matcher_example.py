from getiprompt.data import LVISDataset, FolderDataset
from getiprompt.models import Matcher
from getiprompt.data.base.batch import Batch
from getiprompt.types import Results


from matplotlib import pyplot as plt

def visualize_results(batch, results: Results):
    for img, masks in zip(batch.images, results.masks, strict=False):
        _masks = masks.data
        # make column grid of masks
        col = 5
        row = (len(_masks) + col - 1) // col
        fig, axes = plt.subplots(row, col, figsize=(10, 10))
        for i, (class_id, mask) in enumerate(_masks):
            axes[i // col, i % col].imshow(mask)
            axes[i // col, i % col].set_title(categories[class_id])
        plt.savefig(f"results/matcher_{i}.png")
        plt.close()


def main():
    # dataset = FolderDataset(
    #     root="library/tests/assets/fss-1000",
    #     categories=["apple", "basketball"],
    #     n_shots=2,
    # )

    dataset = LVISDataset(
        root="~/datasets/lvis",
        categories=["doughnut", "cupcake", "pastry"],
        n_shots=1,
    )

    ref_dataset = dataset.get_reference_dataset()
    ref_batch = Batch.collate(ref_dataset)

    model = Matcher(
        precision="bf16",
        compile_models=False,
        benchmark_inference_speed=False,
        device="cuda",
    )

    model.learn(ref_batch)

    target_dataset = dataset.get_target_dataset()
    target_batch = Batch.collate(target_dataset)
    results = model.infer(target_batch)
    print(results)


if __name__ == "__main__":
    main()