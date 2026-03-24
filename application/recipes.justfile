dataset-base-url := "https://storage.geti.intel.com/instant-learn/datasets/"
dataset-names := "led aquarium candies cards nuts potatoes"

download-datasets target_dir:
    #!/usr/bin/env bash
    DATASET_DIR="{{ target_dir }}/coffee-berries"

    mkdir -p $DATASET_DIR
    for filename in {{ dataset-names }}; do
        dataset_subdir="$DATASET_DIR/$filename"
        if [ -d "$dataset_subdir" ] && [ "$(ls -A "$dataset_subdir")" ]; then
            echo "Dataset subdirectory $dataset_subdir already exists and is not empty. Skipping download."
            continue
        fi
        mkdir -p "$dataset_subdir"
        url="{{ dataset-base-url }}$filename.zip"
        echo "Downloading dataset from $url"
        if ! curl -s "$url" -o "$filename.zip"; then
            echo "Error: Failed to download dataset from $url"
            continue  # proceed with next archive
        fi
        echo "Unpacking $filename to $DATASET_DIR"
        unzip -j -q -o "$filename.zip" -d "$dataset_subdir"
        echo "Removing downloaded archive $filename.zip"
        rm "$filename.zip"
    done
